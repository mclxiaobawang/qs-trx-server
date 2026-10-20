# ruff: noqa: RUF001
import json
import math
import threading
import zipfile
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from freqtrade.rpc.api_server.deps import get_config, get_exchange
from freqtrade.rpc.api_server.trx_diagnostic_reader import diagnostic_reader
from freqtrade.strategy.trx_phase1 import diagnostics
from freqtrade.strategy.trx_phase1.account import AccountBusy, AccountService
from freqtrade.strategy.trx_phase1.candidate_stability import (
    load_stability_source,
    run_stability_study,
)
from freqtrade.strategy.trx_phase1.comparisons import VARIANTS, comparison_groups, result_brief
from freqtrade.strategy.trx_phase1.config import ASSUMPTIONS, ResearchConfig
from freqtrade.strategy.trx_phase1.cross_period import load_cross_period_inputs, run_cross_period
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.environment_study import load_sources, run_environment_study
from freqtrade.strategy.trx_phase1.experiments import GROUPS, run_experiment
from freqtrade.strategy.trx_phase1.funding_study import run_funding_study
from freqtrade.strategy.trx_phase1.ledger import StateStore
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import local_execution_status
from freqtrade.strategy.trx_phase1.live_service import LiveService, LiveSetup
from freqtrade.strategy.trx_phase1.protocol import protocol_parameters, run_protocol
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.risk_study import load_risk_source, run_risk_study
from freqtrade.strategy.trx_phase1.runtime import PaperRuntime
from freqtrade.strategy.trx_phase1.shadow import (
    DEFAULT_EXPERIMENT,
    DEFAULT_LABEL,
    DEFAULT_RUN,
    ShadowRuntime,
    create_shadow,
)
from freqtrade.strategy.trx_phase1.shadow_execution import check_execution_faults
from freqtrade.strategy.trx_phase1.studies import list_studies, run_study
from freqtrade.strategy.trx_phase1.timerange import TimeRange


router = APIRouter(prefix="/trx", tags=["TRX research"])
_jobs = {}
_lock = threading.RLock()
_runtime: PaperRuntime | None = None
_account_services: dict[Path, AccountService] = {}
_shadow_runtimes: dict[Path, ShadowRuntime] = {}
_live_services: dict[Path, LiveService] = {}
_live_restorer = None


def shadow_root(config):
    paths(config)
    return Path(config["user_data_dir"]) / "trx_phase1/shadows"


def get_shadow(config, shadow_id):
    path = shadow_root(config) / str(shadow_id) / "shadow.sqlite"
    if not path.is_file():
        raise HTTPException(404, "Shadow account not found")
    with _lock:
        if path not in _shadow_runtimes:
            _shadow_runtimes[path] = ShadowRuntime(path)
        return _shadow_runtimes[path]


@router.get("/shadows")
def shadows(config=Depends(get_config)):
    root = shadow_root(config)
    source = saved_result_path(config, DEFAULT_RUN, experiment_id=DEFAULT_EXPERIMENT)
    parameters = json.loads(source.read_text())["config"]
    report = root / "fault-check.json"
    return {
        "accounts": [
            get_shadow(config, UUID(path.parent.name)).status(curve=False)
            for path in sorted(root.glob("*/shadow.sqlite"), reverse=True)
        ],
        "candidate": {
            "label": DEFAULT_LABEL,
            "run_id": DEFAULT_RUN,
            "experiment_id": DEFAULT_EXPERIMENT,
            "parameters": parameters,
        },
        "fault_checks": json.loads(report.read_text()) if report.exists() else None,
    }


@router.post("/shadows")
def new_shadow(config=Depends(get_config)):
    # Explicit selected candidate only. New-backtest forms cannot alter this snapshot.
    source = saved_result_path(config, DEFAULT_RUN, experiment_id=DEFAULT_EXPERIMENT)
    seed = Path(config["user_data_dir"]) / (
        "trx_phase1/control_matrices/c1-seven-earn-oct16-window-2026-10-16/"
        "inputs/history-through-2026-10-16.feather"
    )
    if not seed.exists():
        seed, _ = paths(config)
    try:
        shadow_id = create_shadow(shadow_root(config), source, seed)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return get_shadow(config, UUID(shadow_id)).status()


@router.post("/shadows/fault-check")
def shadow_fault_check(config=Depends(get_config)):
    source = saved_result_path(config, DEFAULT_RUN, experiment_id=DEFAULT_EXPERIMENT)
    cfg = ResearchConfig.from_saved(json.loads(source.read_text())["config"])
    with _lock:
        return check_execution_faults(shadow_root(config), cfg)


@router.get("/shadows/{shadow_id}")
def shadow_status(shadow_id: UUID, config=Depends(get_config)):
    return get_shadow(config, shadow_id).status()


@router.post("/shadows/{shadow_id}/start")
def start_shadow(shadow_id: UUID, config=Depends(get_config)):
    runtime = get_shadow(config, shadow_id)
    try:
        runtime.start()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return runtime.status()


@router.post("/shadows/{shadow_id}/stop")
def stop_shadow(shadow_id: UUID, config=Depends(get_config)):
    runtime = get_shadow(config, shadow_id)
    runtime.stop()
    return runtime.status()


def get_account_service(config=Depends(get_config)):
    directory = Path(config["user_data_dir"]).resolve()
    with _lock:
        if directory not in _account_services:
            _account_services[directory] = AccountService(directory)
        return _account_services[directory]


@router.get("/account")
def account_status(service=Depends(get_account_service)):
    """Local status only; this GET never triggers a private exchange request."""
    return service.status()


@router.get("/execution")
def live_execution_status(config=Depends(get_config)):
    """Read durable workflow evidence; this route never invokes an exchange client."""
    return local_execution_status(Path(config["user_data_dir"]) / "trx_phase1")


@router.post("/account/refresh")
def refresh_account(service=Depends(get_account_service)):
    """Explicitly read the account; no financial write endpoint exists here."""
    try:
        return service.refresh()
    except AccountBusy as exc:
        raise HTTPException(409, str(exc)) from None


def get_live_service(config=Depends(get_config)):
    directory = Path(config["user_data_dir"]).resolve()
    with _lock:
        if directory not in _live_services:
            _live_services[directory] = LiveService(
                directory,
                existing_only=config.get("trx_phase1", {}).get("existing_live_only", False),
            )
        return _live_services[directory]


def start_live_restore(config):
    global _live_restorer
    settings = config.get("trx_phase1", {})
    if not settings.get("auto_resume_live"):
        return
    if not settings.get("existing_live_only"):
        raise RuntimeError("Automatic live recovery requires existing-live-only mode")
    from freqtrade.strategy.trx_phase1.live_restore import ExistingLiveRestore

    _live_restorer = ExistingLiveRestore(get_live_service(config))
    _live_restorer.start()


def shutdown_live_services():
    global _live_restorer
    if _live_restorer is not None:
        _live_restorer.stop()
        _live_restorer = None
    for service in list(_live_services.values()):
        service.shutdown()


def live_call(method, *args):
    try:
        return method(*args)
    except ExecutionError as exc:
        raise HTTPException(409, str(exc)) from None
    except Exception:
        raise HTTPException(409, "live_operation_failed") from None


class LiveStartRequest(BaseModel):
    review_hash: str = Field(min_length=64, max_length=64, pattern="^[a-f0-9]+$")


class LiveReconcileRequest(BaseModel):
    operation_id: str | None = Field(default=None, max_length=128)
    exchange_id: str | None = Field(default=None, max_length=128)


@router.get("/live")
def live_status(service=Depends(get_live_service), account=Depends(get_account_service)):
    """Local status and frozen parameters only; never starts a worker."""
    snapshot, account_id = account.allocation_context()
    return live_call(service.status, snapshot, account_id)


@router.post("/live/products")
def live_products(service=Depends(get_live_service)):
    return live_call(service.products)


@router.post("/live/prepare")
def live_prepare(body: LiveSetup, service=Depends(get_live_service)):
    """Read-only preflight and local allocation; no exchange writes."""
    return live_call(service.prepare, body)


@router.post("/live/prepare-account")
def live_prepare_account(service=Depends(get_live_service), account=Depends(get_account_service)):
    """Resolve the current account draft on the server, without financial writes."""
    status, account_id = account.allocation_context()
    observed = (status.get("snapshot") or {}).get("observed_at", 0)
    if status.get("stale") or status.get("error") or service.clock() - observed >= 5000:
        refresh_account(account)
        status, account_id = account.allocation_context()
    return live_call(service.prepare_account, status, account_id)


@router.post("/live/start")
def live_start(body: LiveStartRequest, service=Depends(get_live_service)):
    """Explicit launch of the reviewed account/capital/snapshot and its Earn flows."""
    return live_call(service.start, body.review_hash)


@router.post("/live/pause")
def live_pause(service=Depends(get_live_service)):
    return live_call(service.pause)


@router.post("/live/contribution/preview")
def live_contribution_preview(service=Depends(get_live_service)):
    """Reconcile and preview external assets; never place an order or adopt assets."""
    return live_call(service.contribution)


@router.post("/live/contribution/confirm")
def live_contribution_confirm(body: LiveStartRequest, service=Depends(get_live_service)):
    """Operator confirmation adds reviewed assets to the existing strategy capital."""
    return live_call(service.contribution, body.review_hash)


@router.post("/live/reconcile")
def live_reconcile(body: LiveReconcileRequest, service=Depends(get_live_service)):
    """Query existing financial evidence only; never submits or retries a write."""
    return live_call(service.reconcile, body.operation_id, body.exchange_id)


class ReplayRequest(BaseModel):
    parameters: ResearchConfig = Field(
        default_factory=lambda: ResearchConfig(earn_model="net_benefit_hourly")
    )
    timerange: str = ""
    compare_variants: bool = False
    collect_diagnostics: bool = True


class StudyRequest(BaseModel):
    parameters: ResearchConfig
    timerange: str = ""


class PaperAdditionsRequest(BaseModel):
    enabled: bool


class PaperExitRequest(BaseModel):
    event_id: UUID
    source: Literal["principal", "interest"]
    quantity: float = Field(gt=0)
    accept_loss: bool = False


def paths(config):
    if config["exchange"]["name"] != "bitget" or str(config.get("trading_mode", "spot")) != "spot":
        raise HTTPException(400, "TRX research currently requires Bitget spot")
    return Path(config["datadir"]) / "TRX_USDT-1m.feather", Path(
        config["user_data_dir"]
    ) / "trx_phase1/results"


@router.get("/config")
def research_config(config=Depends(get_config)):
    data_path, directory = paths(config)
    data = pd.read_feather(data_path)
    results = []
    if directory.exists():
        for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                UUID(path.stem)
            except ValueError:
                continue
            saved = json.loads(path.read_text())
            saved["run_id"] = path.stem
            results.append(saved)
    return {
        "parameters": ResearchConfig(earn_model="net_benefit_hourly").model_dump(),
        "assumptions": ASSUMPTIONS,
        "data": {
            "rows": len(data),
            "start_utc": str(data.date.min()),
            "end_utc": str(data.date.max()),
        },
        "results": [result_brief(saved) for saved in results[:30]],
        "diagnostic_results": [
            {
                "run_id": saved["run_id"],
                "variant": saved["config"]["variant"],
                "start_utc": saved["start_utc"],
                "end_utc": saved["end_utc"],
            }
            for saved in results
            if saved.get("diagnostics", {}).get("schema") in diagnostics.SUPPORTED_SCHEMAS
        ][:30],
        "comparisons": comparison_groups(results)[:20],
        "studies": list_studies(Path(config["user_data_dir"]) / "trx_phase1/studies"),
        "experiments": list_experiments(config),
        "treasury": {
            "real_account_active": False,
            "implemented": [
                "verified product tiers",
                "locked asset accounting",
                "redemption arrival",
                "credited interest",
                "net USDGO switch forecast",
                "Bitget Classic Savings adapter",
            ],
            "pending_verification": [
                "account/API permissions",
                "TRX/USDT/USDGO product eligibility and quotas",
                "actual interest and redemption rules",
                "USDGO executable FX and historical product data",
            ],
        },
    }


def update_job(run_id, **fields):
    with _lock:
        _jobs[run_id].update(fields)


def run_replay(run_id, request, config):
    try:
        data_path, directory = paths(config)
        data = pd.read_feather(data_path)
        if request.timerange:
            window = TimeRange.parse_timerange(request.timerange)
            if window.startdt:
                data = data.loc[data.date >= window.startdt]
            if window.stopdt:
                data = data.loc[data.date < window.stopdt]
        # Freeze the file once for all variants, including while paper data updates.
        data = data.reset_index(drop=True)
        directory.mkdir(parents=True, exist_ok=True)
        variants = VARIANTS if request.compare_variants else (request.parameters.variant,)
        result_ids = []
        for index, variant in enumerate(variants):
            result_id = str(uuid4()) if request.compare_variants else run_id
            parameters = ResearchConfig(**{**request.parameters.model_dump(), "variant": variant})
            result = replay(
                data,
                parameters,
                progress=lambda value, step=index: update_job(
                    run_id, progress=(step + value) / len(variants)
                ),
                diagnostic_path=directory / f"{result_id}.diagnostics.sqlite"
                if request.collect_diagnostics
                else None,
            )
            result["run_id"] = result_id
            if request.compare_variants:
                result["comparison_id"] = run_id
            temporary = directory / f"{result_id}.json.tmp"
            temporary.write_text(json.dumps(result, allow_nan=False))
            temporary.replace(directory / f"{result_id}.json")
            result_ids.append(result_id)
        with _lock:
            _jobs[run_id].update(
                status="success",
                progress=1,
                summary=result["summary"],
                result_ids=result_ids,
                comparison_id=run_id if request.compare_variants else None,
            )
    except Exception as exc:
        with _lock:
            _jobs[run_id].update(status="failed", error=str(exc))


@router.post("/backtest")
def start_replay(request: ReplayRequest, tasks: BackgroundTasks, config=Depends(get_config)):
    paths(config)
    with _lock:
        if any(job["status"] == "running" for job in _jobs.values()):
            raise HTTPException(409, "A TRX research replay is already running")
        run_id = str(uuid4())
        _jobs[run_id] = {"run_id": run_id, "status": "running", "progress": 0, "error": None}
    tasks.add_task(run_replay, run_id, request, config)
    with _lock:
        return dict(_jobs[run_id])


@router.get("/jobs/{run_id}")
def job_status(run_id: UUID):
    with _lock:
        job = _jobs.get(str(run_id))
        if job is None:
            raise HTTPException(404, "Research job not found")
        return dict(job)


def run_validation(run_id, request, config):
    try:
        data_path, _ = paths(config)
        data = pd.read_feather(data_path)
        if request.timerange:
            window = TimeRange.parse_timerange(request.timerange)
            if window.startdt:
                data = data.loc[data.date >= window.startdt]
            if window.stopdt:
                data = data.loc[data.date < window.stopdt]
        run_study(
            data.reset_index(drop=True),
            request.parameters,
            Path(config["user_data_dir"]) / "trx_phase1/studies",
            run_id,
            progress=lambda value: update_job(run_id, progress=value),
        )
        update_job(run_id, status="success", progress=1, study_id=run_id)
    except Exception as exc:
        update_job(run_id, status="failed", error=str(exc))


@router.post("/studies")
def start_study(request: StudyRequest, tasks: BackgroundTasks, config=Depends(get_config)):
    with _lock:
        if any(job["status"] == "running" for job in _jobs.values()):
            raise HTTPException(409, "Research is already running")
        run_id = str(uuid4())
        _jobs[run_id] = {"run_id": run_id, "status": "running", "progress": 0, "error": None}
    tasks.add_task(run_validation, run_id, request, config)
    return dict(_jobs[run_id])


@router.get("/studies/{study_id}")
def study_result(study_id: UUID, config=Depends(get_config)):
    path = Path(config["user_data_dir"]) / "trx_phase1/studies" / str(study_id) / "manifest.json"
    if not path.exists():
        raise HTTPException(404, "Study is not complete or does not exist")
    return json.loads(path.read_text())


@router.get("/studies/{study_id}/results/{run_id}")
def study_detail(study_id: UUID, run_id: UUID, config=Depends(get_config)):
    path = (
        Path(config["user_data_dir"])
        / "trx_phase1/studies"
        / str(study_id)
        / "results"
        / f"{run_id}.json"
    )
    if not path.exists():
        raise HTTPException(404, "Study detail does not exist")
    return json.loads(path.read_text())


@router.get("/results/{run_id}")
def result(run_id: UUID, config=Depends(get_config)):
    _, directory = paths(config)
    path = directory / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(404, "Result not found")
    return json.loads(path.read_text())


def saved_result_path(config, run_id, study_id=None, experiment_id=None):
    root = Path(config["user_data_dir"]) / "trx_phase1"
    if study_id:
        path = root / "studies" / str(study_id) / "results" / f"{run_id}.json"
    elif experiment_id:
        path = root / "experiments" / str(experiment_id) / "results" / f"{run_id}.json"
    else:
        path = root / "results" / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(404, "Result not found")
    return path


@router.get("/results/{run_id}/diagnostics")
def diagnostic_overview(
    run_id: UUID,
    study_id: UUID | None = None,
    experiment_id: UUID | None = None,
    timezone: Literal["UTC", "Asia/Shanghai"] = "Asia/Shanghai",
    config=Depends(get_config),
):
    path = saved_result_path(config, run_id, study_id, experiment_id).with_suffix(
        ".diagnostics.sqlite"
    )
    if not diagnostics.available(path):
        return {"collected": False, "message": "未采集，需要重跑"}
    try:
        return diagnostic_reader.read(path, timezone)
    except TimeoutError as exc:
        raise HTTPException(504, "诊断汇总尚未完成，请稍后重试；已保存的结果不会改变。") from exc


@router.get("/results/{run_id}/diagnostics/{kind}")
def diagnostic_records(
    run_id: UUID,
    kind: Literal[
        "samples",
        "opportunities",
        "evaluations",
        "fills",
        "lots",
        "drawdowns",
        "operations",
        "period_switches",
        "anchors",
        "conversions",
        "fx_cycles",
    ],
    study_id: UUID | None = None,
    experiment_id: UUID | None = None,
    start: int | None = None,
    stop: int | None = None,
    side: Literal["buy", "sell"] | None = None,
    status: str | None = None,
    reason: str | None = None,
    offset: int = 0,
    limit: int = 100,
    config=Depends(get_config),
):
    if not 1 <= limit <= 2000 or offset < 0:
        raise HTTPException(422, "Invalid page size or offset")
    path = saved_result_path(config, run_id, study_id, experiment_id).with_suffix(
        ".diagnostics.sqlite"
    )
    if not diagnostics.available(path):
        return {"collected": False, "message": "未采集，需要重跑", "rows": []}
    with diagnostics.connect(path) as db:
        if kind == "samples":
            return diagnostics.sample_rows(db, start, stop, limit)
        page = diagnostics.records(db, kind, start, stop, side, status, reason, offset, limit)
        if kind == "lots":
            add_original_lot_costs(db, page["rows"])
        if kind == "operations":
            ids = {row.get("opportunity_id") for row in page["rows"] if row.get("opportunity_id")}
            if ids:
                related = {
                    row["id"]: row
                    for row in (
                        json.loads(value[0])
                        for value in db.execute(
                            "SELECT payload FROM records WHERE kind='opportunities' AND id IN ("
                            + ",".join("?" for _ in ids)
                            + ")",
                            list(ids),
                        )
                    )
                }
                for row in page["rows"]:
                    signal = related.get(row.get("opportunity_id"))
                    if signal:
                        row["signal_outcome"] = {
                            key: signal.get(key)
                            for key in ("id", "side", "status", "reason", "ended_at")
                        }
        return page


def add_original_lot_costs(db, lots):
    """Derive purchase costs from recorded allocation costs, including credited refunds."""
    costs = {}
    for fill in diagnostics.records(db, "fills", limit=1000000)["rows"]:
        for allocation in fill["allocations"]:
            key = allocation["lot_id"]
            costs[key] = costs.get(key, 0) + allocation["cost"]
    for lot in lots:
        cost = lot["remaining_cost"] + costs.get(lot["lot_id"], 0)
        lot["original_cost"] = cost
        lot["original_unit_cost"] = (
            cost / lot["original_quantity"] if lot["original_quantity"] else None
        )
        lot["purchase_cost_basis"] = (
            "recorded sold allocation costs plus remaining cost, net of credited purchase rebates"
        )
    return lots


@router.get("/results/{run_id}/export")
def export_research(
    run_id: UUID,
    study_id: UUID | None = None,
    experiment_id: UUID | None = None,
    config=Depends(get_config),
):
    path = saved_result_path(config, run_id, study_id, experiment_id)
    saved = json.loads(path.read_text())
    target = path.parent / "exports" / f"{run_id}-{diagnostics.SCHEMA}-report1.zip"
    if not target.exists():
        trace = path.with_suffix(".diagnostics.sqlite")
        temporary = target.with_name(target.name + "." + str(uuid4()) + ".tmp")
        diagnostics.export_package(saved, trace, temporary)
        if diagnostics.available(trace):
            with diagnostics.connect(trace) as db:
                lots = diagnostics.records(db, "lots", limit=1000000)["rows"]
                add_original_lot_costs(db, lots)
            with zipfile.ZipFile(temporary, "a", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("principal_lot_costs.json", diagnostics.encoded(lots))
        temporary.replace(target)
    return FileResponse(target, media_type="application/zip", filename=f"trx-research-{run_id}.zip")


@router.get("/results/{run_id}/interval")
def diagnostic_interval(
    run_id: UUID,
    start: int,
    stop: int,
    study_id: UUID | None = None,
    experiment_id: UUID | None = None,
    config=Depends(get_config),
):
    if stop <= start:
        raise HTTPException(422, "Invalid interval")
    path = saved_result_path(config, run_id, study_id, experiment_id).with_suffix(
        ".diagnostics.sqlite"
    )
    if not diagnostics.available(path):
        return {"collected": False, "message": "未采集，需要重跑"}
    return diagnostics.interval_details(path, start, stop)


class SavedReference(BaseModel):
    run_id: UUID
    study_id: UUID | None = None
    experiment_id: UUID | None = None


def frozen_reference(config, reference):
    from freqtrade.strategy.trx_phase1.candles import data_fingerprint, validate_minutes

    path = saved_result_path(config, reference.run_id, reference.study_id, reference.experiment_id)
    saved = json.loads(path.read_text())
    data = pd.read_feather(paths(config)[0])
    data = data.loc[
        (data.date >= pd.Timestamp(saved["start_utc"]))
        & (data.date <= pd.Timestamp(saved["end_utc"]))
    ].reset_index(drop=True)
    data = validate_minutes(data)
    if data_fingerprint(data) != saved.get("data_fingerprint"):
        raise ValueError("原始行情指纹不一致，无法进行同数据重跑")
    return saved, data


def assert_reproduced(saved, replayed):
    # Old versions can omit newer summary fields; compare the fields actually saved.
    for key in ("orders", "fills", "equity_curve"):
        if key in saved and saved[key] != replayed[key]:
            raise ValueError(f"当前规则无法复现历史结果（{key}）；未发布为同版本诊断")
    for key in (
        "equity",
        "total_profit",
        "return_ratio",
        "max_drawdown",
        "buy_orders",
        "sell_orders",
    ):
        if key in saved["summary"] and saved["summary"][key] != replayed["summary"][key]:
            raise ValueError(f"当前规则无法复现历史结果（{key}）")


def run_diagnostic_replay(run_id, reference, config):
    try:
        saved, data = frozen_reference(config, reference)
        directory = paths(config)[1]
        directory.mkdir(parents=True, exist_ok=True)
        trace = directory / f"{run_id}.diagnostics.sqlite"
        replayed = replay(
            data,
            ResearchConfig.from_saved(saved["config"]),
            progress=lambda value: update_job(run_id, progress=value),
            diagnostic_path=trace,
        )
        try:
            assert_reproduced(saved, replayed)
        except Exception:
            trace.unlink(missing_ok=True)
            raise
        replayed.update(run_id=run_id, diagnostic_reference=reference.model_dump(mode="json"))
        temp = directory / f"{run_id}.json.tmp"
        temp.write_text(json.dumps(replayed, allow_nan=False))
        temp.replace(directory / f"{run_id}.json")
        update_job(run_id, status="success", progress=1, result_ids=[run_id])
    except Exception as exc:
        update_job(run_id, status="failed", error=str(exc))


@router.post("/diagnostics/replay")
def rerun_diagnostics(
    reference: SavedReference, tasks: BackgroundTasks, config=Depends(get_config)
):
    saved_result_path(config, reference.run_id, reference.study_id, reference.experiment_id)
    with _lock:
        if any(job["status"] == "running" for job in _jobs.values()):
            raise HTTPException(409, "Research is already running")
        run_id = str(uuid4())
        _jobs[run_id] = {"run_id": run_id, "status": "running", "progress": 0, "error": None}
    tasks.add_task(run_diagnostic_replay, run_id, reference, config)
    return dict(_jobs[run_id])


class ExperimentRequest(SavedReference):
    group: Literal[
        "buy_threshold",
        "sell_threshold",
        "buy_fraction",
        "sell_fraction",
        "buy_quantity_dynamic",
        "sell_quantity_dynamic",
        "disable_earn",
        "execution_preflight",
        "buy_reference_recovery",
        "sell_quantity_basis",
        "sell_fraction_cap",
    ]
    values: list[float | bool | str] = Field(min_length=1, max_length=5)


def rate_snapshot_path(config):
    return (
        Path(config["user_data_dir"]) / "trx_phase1/research_inputs/bitget-earn-apr-2026-10-13.json"
    )


@router.get("/protocol/parameters")
def focused_parameters(
    run_id: UUID,
    study_id: UUID | None = None,
    experiment_id: UUID | None = None,
    config=Depends(get_config),
):
    saved = json.loads(saved_result_path(config, run_id, study_id, experiment_id).read_text())
    if not rate_snapshot_path(config).is_file():
        raise HTTPException(404, "Public APR snapshot is unavailable")
    parameters, snapshot = protocol_parameters(saved, rate_snapshot_path(config))
    return {"parameters": parameters.model_dump(), "snapshot": snapshot}


def run_focused(run_id, reference, config):
    try:
        saved, data = frozen_reference(config, reference)
        run_protocol(
            data,
            saved,
            Path(config["user_data_dir"]) / "trx_phase1/experiments",
            run_id,
            rate_snapshot_path(config),
            progress=lambda value: update_job(run_id, progress=value),
        )
        update_job(run_id, status="success", progress=1, experiment_id=run_id)
    except Exception as exc:
        update_job(run_id, status="failed", error=str(exc))


@router.post("/protocol")
def start_focused(reference: SavedReference, tasks: BackgroundTasks, config=Depends(get_config)):
    saved_result_path(config, reference.run_id, reference.study_id, reference.experiment_id)
    if not rate_snapshot_path(config).is_file():
        raise HTTPException(404, "Public APR snapshot is unavailable")
    with _lock:
        if any(job["status"] == "running" for job in _jobs.values()):
            raise HTTPException(409, "Research is already running")
        run_id = str(uuid4())
        _jobs[run_id] = {"run_id": run_id, "status": "running", "progress": 0, "error": None}
    tasks.add_task(run_focused, run_id, reference, config)
    return dict(_jobs[run_id])


class MatchedEnvironmentRequest(BaseModel):
    reference_experiment_id: UUID
    legacy_study_id: UUID | None = None


def run_matched_environment(run_id, request, config):
    try:
        root = Path(config["user_data_dir"]) / "trx_phase1"
        reference, legacy, originals = load_sources(
            root, request.reference_experiment_id, request.legacy_study_id
        )
        anchor = next(row for row in reference["entries"] if row["case"] == "buy_08")
        _, frame = frozen_reference(
            config,
            SavedReference(run_id=anchor["run_id"], experiment_id=reference["experiment_id"]),
        )
        run_environment_study(
            frame,
            reference,
            legacy,
            originals,
            root,
            run_id,
            assert_reproduced,
            progress=lambda value: update_job(run_id, progress=value),
        )
        update_job(run_id, status="success", progress=1, experiment_id=run_id)
    except Exception as exc:
        update_job(run_id, status="failed", error=str(exc))


@router.post("/legacy-environment")
def start_matched_environment(
    request: MatchedEnvironmentRequest, tasks: BackgroundTasks, config=Depends(get_config)
):
    root = Path(config["user_data_dir"]) / "trx_phase1"
    try:
        load_sources(root, request.reference_experiment_id, request.legacy_study_id)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc
    with _lock:
        if any(job["status"] == "running" for job in _jobs.values()):
            raise HTTPException(409, "Research is already running")
        run_id = str(uuid4())
        _jobs[run_id] = {"run_id": run_id, "status": "running", "progress": 0, "error": None}
    tasks.add_task(run_matched_environment, run_id, request, config)
    return dict(_jobs[run_id])


@router.get("/experiments/{experiment_id}/benchmarks/{scenario}/{name}")
def benchmark_csv(
    experiment_id: UUID,
    scenario: Literal["ordinary_snapshot", "conservative_pressure", "continuous_history"],
    name: Literal[
        "usdt",
        "usdgo",
        "trx_hold",
        "low_exposure",
        "exposure_40",
        "exposure_70",
        "half_hold",
        "allocation_10",
        "allocation_20",
        "allocation_30",
        "allocation_40",
        "allocation_50",
        "allocation_60",
        "allocation_70",
        "allocation_80",
        "allocation_90",
    ],
    config=Depends(get_config),
):
    path = (
        Path(config["user_data_dir"])
        / "trx_phase1/experiments"
        / str(experiment_id)
        / "benchmarks"
        / scenario
        / f"{name}.csv"
    )
    if not path.is_file():
        raise HTTPException(404, "Full benchmark curve was not collected")
    return FileResponse(path, media_type="text/csv", filename=f"{scenario}-{name}-minute-nav.csv")


class TwoTrackRequest(BaseModel):
    source_experiment_id: UUID


def two_track_source(config, request):
    root = Path(config["user_data_dir"]) / "trx_phase1"
    path = root / "experiments" / str(request.source_experiment_id) / "manifest.json"
    manifest = json.loads(path.read_text())
    source_id = (
        manifest["source_experiment_id"]
        if manifest["group"] in {"risk_matched_rules", "funding_preparation", "candidate_stability"}
        else request.source_experiment_id
    )
    return root, *load_risk_source(root, source_id)


def run_two_track(run_id, request, config, funding):
    try:
        root, source, saved = two_track_source(config, request)
        anchor = saved["legacy_dynamic_quantity" if funding else "inventory_08"]
        _, frame = frozen_reference(
            config, SavedReference(run_id=anchor["run_id"], experiment_id=source["experiment_id"])
        )
        runner = run_funding_study if funding else run_risk_study
        runner(
            frame,
            source,
            anchor if funding else saved,
            root,
            run_id,
            assert_reproduced,
            progress=lambda value: update_job(run_id, progress=value),
        )
        update_job(run_id, status="success", progress=1, experiment_id=run_id)
    except Exception as exc:
        update_job(run_id, status="failed", error=str(exc))


def start_two_track(request, tasks, config, funding):
    try:
        two_track_source(config, request)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc
    with _lock:
        if any(job["status"] == "running" for job in _jobs.values()):
            raise HTTPException(409, "Research is already running")
        run_id = str(uuid4())
        _jobs[run_id] = {"run_id": run_id, "status": "running", "progress": 0, "error": None}
    tasks.add_task(run_two_track, run_id, request, config, funding)
    return dict(_jobs[run_id])


@router.post("/risk-study")
def start_risk_study(request: TwoTrackRequest, tasks: BackgroundTasks, config=Depends(get_config)):
    return start_two_track(request, tasks, config, False)


@router.post("/funding-study")
def start_funding_study(
    request: TwoTrackRequest, tasks: BackgroundTasks, config=Depends(get_config)
):
    return start_two_track(request, tasks, config, True)


def run_candidate_stability(run_id, request, config):
    try:
        root = Path(config["user_data_dir"]) / "trx_phase1"
        source, saved = load_stability_source(root, request.source_experiment_id)
        _, frame = frozen_reference(
            config,
            SavedReference(run_id=saved["C1"]["run_id"], experiment_id=source["experiment_id"]),
        )
        run_stability_study(
            frame,
            source,
            saved,
            root,
            run_id,
            assert_reproduced,
            progress=lambda value: update_job(run_id, progress=value),
        )
        update_job(run_id, status="success", progress=1, experiment_id=run_id)
    except Exception as exc:
        update_job(run_id, status="failed", error=str(exc))


@router.post("/candidate-stability")
def start_candidate_stability(
    request: TwoTrackRequest, tasks: BackgroundTasks, config=Depends(get_config)
):
    try:
        load_stability_source(
            Path(config["user_data_dir"]) / "trx_phase1", request.source_experiment_id
        )
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc
    with _lock:
        if any(job["status"] == "running" for job in _jobs.values()):
            raise HTTPException(409, "Research is already running")
        run_id = str(uuid4())
        _jobs[run_id] = {"run_id": run_id, "status": "running", "progress": 0, "error": None}
    tasks.add_task(run_candidate_stability, run_id, request, config)
    return dict(_jobs[run_id])


def run_cross_period_job(run_id, request, config):
    try:
        root = Path(config["user_data_dir"]) / "trx_phase1"
        run_cross_period(
            root,
            request.source_experiment_id,
            run_id,
            assert_reproduced,
            progress=lambda value: update_job(run_id, progress=value),
        )
        update_job(run_id, status="success", progress=1, experiment_id=run_id)
    except Exception as exc:
        update_job(run_id, status="failed", error=str(exc))


@router.post("/cross-period-validation")
def start_cross_period_validation(
    request: TwoTrackRequest, tasks: BackgroundTasks, config=Depends(get_config)
):
    try:
        load_cross_period_inputs(
            Path(config["user_data_dir"]) / "trx_phase1", request.source_experiment_id
        )
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc
    with _lock:
        if any(job["status"] == "running" for job in _jobs.values()):
            raise HTTPException(409, "Research is already running")
        run_id = str(uuid4())
        _jobs[run_id] = {"run_id": run_id, "status": "running", "progress": 0, "error": None}
    tasks.add_task(run_cross_period_job, run_id, request, config)
    return dict(_jobs[run_id])


def list_experiments(config):
    root = Path(config["user_data_dir"]) / "trx_phase1/experiments"
    manifests = sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            **{
                key: saved[key]
                for key in ("experiment_id", "group", "start_utc", "end_utc", "stage")
            },
            **({"title": saved["title"]} if saved.get("title") else {}),
        }
        for saved in (json.loads(path.read_text()) for path in manifests[:20])
    ]


def run_controlled(run_id, request, config):
    try:
        saved, data = frozen_reference(config, request)
        run_experiment(
            data,
            saved,
            Path(config["user_data_dir"]) / "trx_phase1/experiments",
            run_id,
            request.group,
            request.values,
            assert_reproduced,
            progress=lambda value: update_job(run_id, progress=value),
        )
        update_job(run_id, status="success", progress=1, experiment_id=run_id)
    except Exception as exc:
        update_job(run_id, status="failed", error=str(exc))


@router.post("/experiments")
def start_experiment(
    request: ExperimentRequest, tasks: BackgroundTasks, config=Depends(get_config)
):
    if request.group not in GROUPS:
        raise HTTPException(422, "Unsupported experiment group")
    saved_result_path(config, request.run_id, request.study_id, request.experiment_id)
    with _lock:
        if any(job["status"] == "running" for job in _jobs.values()):
            raise HTTPException(409, "Research is already running")
        run_id = str(uuid4())
        _jobs[run_id] = {"run_id": run_id, "status": "running", "progress": 0, "error": None}
    tasks.add_task(run_controlled, run_id, request, config)
    return dict(_jobs[run_id])


def saved_experiment_fills(entry, directory):
    try:
        run_id = UUID(entry["run_id"])
        owner = UUID(entry.get("source_experiment_id", directory.name))
    except (KeyError, ValueError, TypeError):
        return None
    result_path = directory.parent / str(owner) / "results" / f"{run_id}.json"
    diagnostic_path = result_path.with_suffix(".diagnostics.sqlite")
    if diagnostic_path.is_file() and diagnostics.available(diagnostic_path):
        with diagnostics.connect(diagnostic_path) as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT payload FROM records WHERE kind='fills'")
            ]
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        return result.get("fills", result.get("orders"))
    return None


def add_trading_totals(saved, directory):
    """Read actual execution totals without rewriting frozen research artifacts."""

    def numeric(value):
        return type(value) in (int, float) and math.isfinite(value)

    for entry in saved.get("entries", []):
        if "summary" not in entry:
            continue
        summary = entry["summary"]
        amounts = entry.get("amounts", {})
        totals = [amounts.get(side, {}).get("total") for side in ("buy", "sell")]
        if all(numeric(value) for value in totals):
            summary["trading_turnover"] = sum(totals)
            continue
        fills = saved_experiment_fills(entry, directory)
        if fills is None:
            continue
        if all(numeric(row.get("quantity")) and numeric(row.get("price")) for row in fills):
            summary["trading_turnover"] = math.fsum(row["quantity"] * row["price"] for row in fills)
        if "gross_fees" not in summary and all(numeric(row.get("fee")) for row in fills):
            summary["gross_fees"] = math.fsum(row["fee"] for row in fills)

    for benchmark in saved.get("benchmarks", []):
        summary = benchmark["summary"]
        if benchmark.get("cash_only"):
            summary["trading_turnover"] = 0.0
        elif (
            benchmark.get("execution", "").startswith("one initial purchase")
            and summary.get("buy_orders", 0) > 0
            and summary.get("sell_orders") == 0
            and numeric(summary.get("remaining_cost"))
            and numeric(summary.get("fees"))
        ):
            # With no sales, held cost is the full purchase notional plus fees
            # minus credited rebates. Removing net fees recovers actual turnover.
            summary["trading_turnover"] = summary["remaining_cost"] - summary["fees"]


@router.get("/experiments/{experiment_id}")
def experiment_result(experiment_id: UUID, config=Depends(get_config)):
    path = (
        Path(config["user_data_dir"])
        / "trx_phase1/experiments"
        / str(experiment_id)
        / "manifest.json"
    )
    if not path.exists():
        raise HTTPException(404, "Experiment is incomplete or does not exist")
    saved = json.loads(path.read_text())
    add_trading_totals(saved, path.parent)
    if saved.get("group") in {"funding_preparation", "candidate_stability"}:
        # The overview needs transition balances, not eight copies of complete
        # inherited receipt/eligibility histories. Full snapshots stay in the
        # immutable manifest, individual results and exported research package.
        for row in saved["entries"]:
            transition = row.get("funding_transition")
            if transition:
                transition.pop("earn_state", None)
    if saved.get("group") == "candidate_stability":
        analysis = path.parent / "analysis/funding-after180.json"
        if analysis.is_file():
            saved["funding_after180"] = json.loads(analysis.read_text())
    if saved.get("group") == "cross_period_validation":
        analysis = path.parent / "analysis/same-calendar-audit.json"
        if analysis.is_file():
            saved["same_calendar_audit"] = json.loads(analysis.read_text())
        quality = path.parent / "analysis/price-quality-audit.json"
        if quality.is_file():
            report = json.loads(quality.read_text())
            saved["price_quality_audit"] = {
                "interpretation": report["interpretation"],
                "cases": {
                    name: {
                        "zero_volume_minute_fill_count": row["zero_volume_minute_fill_count"],
                        "tagged_extreme_price_lot": row["tagged_extreme_price_lot"],
                    }
                    for name, row in report["cases"].items()
                },
            }
    return saved


@router.get("/experiments/{experiment_id}/analysis/{name}")
def candidate_analysis_file(
    experiment_id: UUID,
    name: Literal[
        "c1-source-minute-paths.csv",
        "c1-source-monthly.csv",
        "c1-source-batches.csv",
        "c1-attribution.json",
        "funding-after180.json",
        "neighbor-boundaries.json",
        "same-calendar-audit.json",
        "month-end-equity-gaps.csv",
        "price-quality-audit.json",
    ],
    config=Depends(get_config),
):
    path = (
        Path(config["user_data_dir"])
        / "trx_phase1/experiments"
        / str(experiment_id)
        / "analysis"
        / name
    )
    if not path.is_file():
        raise HTTPException(404, "Candidate attribution was not collected")
    return FileResponse(
        path, filename=name, media_type="text/csv" if name.endswith(".csv") else "application/json"
    )


@router.get("/experiments/{experiment_id}/results/{run_id}")
def experiment_detail(experiment_id: UUID, run_id: UUID, config=Depends(get_config)):
    path = saved_result_path(config, run_id, experiment_id=experiment_id)
    return json.loads(path.read_text())


@router.get("/experiments/{experiment_id}/analysis/{run_id}/{name}")
def cross_period_analysis_file(
    experiment_id: UUID,
    run_id: UUID,
    name: Literal["monthly.csv", "annual.csv", "daily-path.csv"],
    config=Depends(get_config),
):
    path = (
        Path(config["user_data_dir"])
        / "trx_phase1/experiments"
        / str(experiment_id)
        / "analysis"
        / str(run_id)
        / name
    )
    if not path.is_file():
        raise HTTPException(404, "Continuous account period report was not collected")
    return FileResponse(path, filename=f"{run_id}-{name}", media_type="text/csv")


@router.get("/paper")
def paper_status(config=Depends(get_config)):
    if _runtime:
        return _runtime.status()
    store = StateStore(Path(config["user_data_dir"]) / "trx_phase1/paper.sqlite")
    _, saved = store.load("paper")
    if saved:
        engine = Engine.restore(saved)
        return {
            "running": False,
            "mode": "paper_only",
            "config": engine.config.model_dump(),
            "summary": engine.status(),
            "orders": engine.book.order_summary(),
            "lots": [lot.state() for lot in engine.book.lots],
            "manual_preparation": engine.manual_preparation,
        }
    return {"running": False, "mode": "paper_only", "summary": None, "orders": [], "lots": []}


@router.post("/paper/start")
def start_paper(parameters: ResearchConfig, config=Depends(get_config)):
    global _runtime
    with _lock:
        if _runtime and _runtime.running:
            raise HTTPException(409, "Paper runtime already running")
        data_path, _ = paths(config)
        try:
            _runtime = PaperRuntime(
                Path(config["user_data_dir"]), data_path, get_exchange(config), parameters
            )
            _runtime.start()
            return _runtime.status()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc


@router.post("/paper/stop")
def stop_paper():
    if _runtime:
        try:
            _runtime.stop()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return _runtime.status()
    return {"running": False, "mode": "paper_only", "summary": None, "orders": [], "lots": []}


def stored_paper(config, requested=None):
    global _runtime
    with _lock:
        if _runtime is None:
            data_path, _ = paths(config)
            store = StateStore(Path(config["user_data_dir"]) / "trx_phase1/paper.sqlite")
            _, saved = store.load("paper")
            parameters = (
                ResearchConfig.from_saved(saved["config"])
                if saved
                else requested or ResearchConfig()
            )
            _runtime = PaperRuntime(
                Path(config["user_data_dir"]), data_path, get_exchange(config), parameters
            )
    return _runtime


@router.post("/paper/configure")
def configure_paper(parameters: ResearchConfig, config=Depends(get_config)):
    try:
        with _lock:
            runtime = stored_paper(config, parameters)
            runtime.configure(parameters)
        return runtime.status()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/paper/additions")
def paper_additions(request: PaperAdditionsRequest, config=Depends(get_config)):
    runtime = stored_paper(config)
    runtime.set_additions(request.enabled)
    return runtime.status()


@router.post("/paper/refresh")
def refresh_paper(config=Depends(get_config)):
    try:
        with _lock:
            runtime = stored_paper(config)
            runtime.refresh_paused()
        return runtime.status()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/paper/exit")
def exit_paper(request: PaperExitRequest, config=Depends(get_config)):
    try:
        with _lock:
            runtime = stored_paper(config)
            runtime.manual_exit(
                str(request.event_id), request.source, request.quantity, request.accept_loss
            )
        return runtime.status()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/paper/prepare-exit")
def prepare_paper_exit(request: PaperExitRequest, config=Depends(get_config)):
    try:
        with _lock:
            runtime = stored_paper(config)
            runtime.prepare_exit(str(request.event_id), request.source, request.quantity)
        return runtime.status()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/paper/cancel-exit")
def cancel_paper_exit(config=Depends(get_config)):
    with _lock:
        runtime = stored_paper(config)
        runtime.cancel_exit()
    return runtime.status()
