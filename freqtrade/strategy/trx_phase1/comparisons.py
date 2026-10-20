import json

from freqtrade.strategy.trx_phase1.config import ResearchConfig


VARIANTS = ("fixed", "dynamic_size", "dynamic_period")


def result_brief(result):
    return {
        "run_id": result["run_id"],
        "variant": result["config"]["variant"],
        "rebate_ratio": result["config"].get("fee_rebate_ratio", 0),
        "summary": result["summary"],
        "start_utc": result["start_utc"],
        "end_utc": result["end_utc"],
    }


def comparison_groups(results):
    """Only complete, identical-data/identical-assumption cohorts may be compared."""
    cohorts = {}
    for result in results:
        if result.get("comparison_id"):
            cohorts.setdefault(result["comparison_id"], []).append(result)
    groups, seen = [], set()
    for comparison_id, cohort in cohorts.items():
        if len(cohort) != 3 or {r["config"]["variant"] for r in cohort} != set(VARIANTS):
            continue
        signatures = []
        for result in cohort:
            settings = ResearchConfig.from_saved(result["config"]).model_dump(exclude={"variant"})
            signatures.append(
                json.dumps(
                    [
                        result["start_utc"],
                        result["end_utc"],
                        result["candles"],
                        result["warmup_minutes"],
                        result.get("data_fingerprint"),
                        settings,
                    ],
                    sort_keys=True,
                )
            )
        if not cohort[0].get("data_fingerprint") or len(set(signatures)) != 1:
            continue
        if signatures[0] in seen:
            continue
        seen.add(signatures[0])
        first = cohort[0]
        by_variant = {result["config"]["variant"]: result for result in cohort}
        groups.append(
            {
                "comparison_id": comparison_id,
                "start_utc": first["start_utc"],
                "end_utc": first["end_utc"],
                "candles": first["candles"],
                "warmup_minutes": first["warmup_minutes"],
                "parameters": ResearchConfig.from_saved(first["config"]).model_dump(
                    exclude={"variant"}
                ),
                "variants": [result_brief(by_variant[variant]) for variant in VARIANTS],
            }
        )
    return groups
