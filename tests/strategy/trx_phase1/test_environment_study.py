import json
from uuid import uuid4

import pytest

from freqtrade.rpc.api_server.api_trx import assert_reproduced
from freqtrade.strategy.trx_phase1.config import MECHANISMS, ResearchConfig
from freqtrade.strategy.trx_phase1.environment_study import (
    CURRENT_CASES,
    ENVIRONMENT_FIELDS,
    load_sources,
    pareto,
    run_environment_study,
    study_cases,
)
from freqtrade.strategy.trx_phase1.replay import replay
from tests.strategy.trx_phase1.test_diagnostics import bars


@pytest.fixture
def sources(tmp_path):
    frame = bars(250)
    legacy_id, reference_id = str(uuid4()), str(uuid4())
    legacy_dir, reference_dir = (
        tmp_path / "studies" / legacy_id,
        tmp_path / "experiments" / reference_id,
    )
    for path in (legacy_dir, reference_dir):
        (path / "results").mkdir(parents=True)

    def save(path, config):
        result = replay(frame, config)
        result["run_id"] = str(uuid4())
        (path / "results" / f"{result['run_id']}.json").write_text(json.dumps(result))
        return result

    old = {}
    for variant in MECHANISMS:
        old[variant] = save(
            legacy_dir,
            ResearchConfig(
                version="trx-phase1-v0.4",
                variant=variant,
                volatility_window=20,
                buy_threshold=0.013,
                earn_model="net_benefit_hourly",
            ),
        )
    profile = ResearchConfig(
        version="trx-phase1-v0.5",
        volatility_window=20,
        earn_model="net_benefit_hourly",
        execution_preflight=True,
        buy_fraction=0.08,
        trx_apr=0.1014,
        usdgo_apr=0.066,
        usdt_apr_tiers=[(300, 0.0737), (120000000, 0.0271)],
        usdgo_apr_tiers=[(100000, 0.066)],
        apr_source="test-frozen-snapshot",
        trx_earn_cap=5000,
        usdgo_earn_cap=1000,
        redemption_delay_minutes=1,
        usdt_redemption_delay_minutes=1,
        fx_fee=0.0005,
        fx_spread=0.0005,
        fx_slippage=0.0002,
    )
    entries = []
    for key, changes in (
        ("buy_06", {"buy_fraction": 0.06}),
        ("buy_08", {}),
        ("inventory_08", {"sell_quantity_basis": "profitable"}),
        ("combined_08", {"sell_quantity_basis": "profitable", "buy_reference_recovery": True}),
    ):
        cfg = ResearchConfig(**{**profile.model_dump(), **changes})
        value = save(reference_dir, cfg)
        entries.append({"case": key, **{k: value[k] for k in ("run_id", "config", "summary")}})
        if key in ("buy_08", "combined_08"):
            value = save(
                reference_dir,
                ResearchConfig(
                    **{
                        **cfg.model_dump(),
                        "apr_scenario_scale": 0.5,
                        "trx_earn_cap": 1000,
                        "usdgo_earn_cap": 500,
                        "redemption_delay_minutes": 60,
                        "usdt_redemption_delay_minutes": 60,
                    }
                ),
            )
            entries.append(
                {
                    "case": "pressure_08" if key == "buy_08" else "pressure_combined",
                    **{k: value[k] for k in ("run_id", "config", "summary")},
                }
            )
    legacy = {
        "study_id": legacy_id,
        "data_fingerprint": old["fixed"]["data_fingerprint"],
        "variants": [
            {"variant": variant, "run_id": saved["run_id"], "summary": saved["summary"]}
            for variant, saved in old.items()
        ],
    }
    reference = {
        "experiment_id": reference_id,
        "group": "focused_protocol",
        "entries": entries,
        "data_fingerprint": old["fixed"]["data_fingerprint"],
        "rate_snapshot": {"test": True},
    }
    (legacy_dir / "manifest.json").write_text(json.dumps(legacy))
    (reference_dir / "manifest.json").write_text(json.dumps(reference))
    reference, legacy, old = json.loads(json.dumps([reference, legacy, old]))
    return frame, tmp_path, reference, legacy, old


def test_common_conditions_and_saved_rule_preservation(sources):
    _, root, reference, legacy, old = sources
    loaded = load_sources(root, reference["experiment_id"])
    assert loaded == (reference, legacy, old)
    cases = study_cases(reference, old)
    assert len(cases) == 24
    for scenario in ("ordinary_snapshot", "conservative_pressure"):
        rows = [row for row in cases if row["scenario"] == scenario]
        assert len(rows) == 12
        common = rows[0]["config"].model_dump()
        assert all(
            all(row["config"].model_dump()[k] == common[k] for k in ENVIRONMENT_FIELDS)
            for row in rows
        )
        assert {row["config"].variant for row in rows if row["family"] == "legacy"} == set(
            MECHANISMS
        )
        for row in rows:
            if row["family"] == "legacy":
                cfg = row["config"]
                assert cfg.buy_threshold == 0.013 and cfg.buy_fraction == 0.12
                assert not cfg.execution_preflight and not cfg.buy_reference_recovery
                original = ResearchConfig.from_saved(old[cfg.variant]["config"]).model_dump()
                assert all(
                    cfg.model_dump()[k] == original[k]
                    for k in original
                    if k not in ENVIRONMENT_FIELDS
                )


def test_wrong_interval_and_incomplete_legacy_study_are_rejected(sources):
    _, root, reference, legacy, _ = sources
    path = root / "studies" / legacy["study_id"] / "manifest.json"
    legacy["data_fingerprint"] = {"wrong": "interval"}
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="exact saved data interval"):
        load_sources(root, reference["experiment_id"])
    legacy["data_fingerprint"] = reference["data_fingerprint"]
    legacy["variants"].pop()
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="complete eight-rule"):
        load_sources(root, reference["experiment_id"])


def test_matched_replays_publish_only_after_current_controls_reproduce(sources):
    frame, root, reference, legacy, old = sources
    before = (root / "experiments" / reference["experiment_id"] / "manifest.json").read_bytes()
    checked = []

    def verify(saved, actual):
        assert_reproduced(saved, actual)
        checked.append(saved["run_id"])

    manifest = run_environment_study(
        frame, reference, legacy, old, root, str(uuid4()), verify, collect_diagnostics=False
    )
    assert len(checked) == len(CURRENT_CASES) + 2
    assert len(manifest["entries"]) == 24 and len(manifest["benchmarks"]) == 8
    by_case = {row["case"]: row for row in manifest["entries"]}
    for row in manifest["entries"]:
        control = by_case[row["compare_to"]] if row["compare_to"] else row
        assert control["scenario"] == row["scenario"]
        assert (
            row["return_change"]
            == row["summary"]["return_ratio"] - control["summary"]["return_ratio"]
        )
    assert (
        root / "experiments" / reference["experiment_id"] / "manifest.json"
    ).read_bytes() == before
    with pytest.raises(ValueError, match="Replay data differs"):
        run_environment_study(frame.iloc[:-1], reference, legacy, old, root, str(uuid4()), verify)


def test_frontier_excludes_cash_and_dominated_paths():
    def row(name, returns, dd, cash=False):
        return {
            "case": name,
            "summary": {"return_ratio": returns, "max_drawdown": dd},
            "cash_only": cash,
        }

    values = [
        row("low-risk", 0.08, 0.03),
        row("high-return", 0.12, 0.07),
        row("dominated", 0.075, 0.06),
        row("equal-path", 0.12, 0.07),
        row("cash-only", 0.2, 0, True),
    ]
    assert pareto(values) == {
        "front": ["low-risk", "high-return", "equal-path"],
        "excluded_cash_only": ["cash-only"],
    }
