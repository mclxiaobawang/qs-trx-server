"""Operator entry point. No command is invoked by the webserver or shadow runtime."""

import argparse
import json
import threading
import time
from pathlib import Path

from freqtrade.strategy.trx_phase1.account import load_settings
from freqtrade.strategy.trx_phase1.live_broker import (
    ClassicBroker,
    ClassicTransport,
    ExecutionError,
)
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from freqtrade.strategy.trx_phase1.request_limits import RequestLimiter


def parser():
    result = argparse.ArgumentParser(description="Classic TRX execution workflows")
    result.add_argument("--root", type=Path, default=Path("user_data/trx_phase1"))
    result.add_argument("--credentials", type=Path)
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Read and allocate an execution ledger; no writes")
    init.add_argument("--capital", required=True)
    init.add_argument("--products", type=Path, required=True)
    for name in ("status", "queue", "poll", "watch", "attach-order", "income", "stop"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--account-id", required=True)
        if name == "queue":
            cmd.add_argument("--plan", type=Path, required=True)
            cmd.add_argument("--ttl", type=int, default=600)
        elif name in {"poll", "watch"}:
            cmd.add_argument(
                "--execute", action="store_true", help="Permit the selected flow's exchange writes"
            )
            if name == "watch":
                cmd.add_argument("--interval", type=float, default=5)
        elif name == "attach-order":
            cmd.add_argument("--operation-id", required=True)
            cmd.add_argument("--exchange-order-id", required=True)
    return result


def main(argv=None):  # noqa: C901 - explicit operator command dispatch
    args = parser().parse_args(argv)
    # Status, queue and attachment are local-only; they cannot contact an exchange.
    local = args.command in {"status", "queue", "attach-order", "stop"}
    try:
        broker = None
        if not local:
            path = args.credentials or args.root / "bitget-account.local.json"
            settings, _, complete = load_settings(path)
            if not complete:
                raise ExecutionError("credentials_unavailable")
            transport = ClassicTransport(
                settings,
                writes_enabled=getattr(args, "execute", False),
                limiter=RequestLimiter(path.parent / "request-limits.sqlite"),
            )
            transport.sync_clock()
            broker = ClassicBroker(transport)
        now = int(time.time() * 1000)
        if args.command == "init":
            runtime = LiveExecution.create(
                args.root, broker, args.capital, json.loads(args.products.read_text()), now
            )
        else:
            runtime = LiveExecution(args.root, broker, args.account_id)
            if runtime.state is None:
                raise ExecutionError("execution_account_not_initialized")
            if args.command == "queue":
                plan = json.loads(args.plan.read_text())
                runtime.queue(
                    plan["steps"],
                    now=now,
                    expires_at=now + args.ttl * 1000,
                    label=plan["label"],
                    strategy_snapshot=plan.get("strategy_snapshot"),
                )
            elif args.command == "attach-order":
                runtime.attach_exchange_order(args.operation_id, args.exchange_order_id, now=now)
            elif args.command == "income":
                runtime.import_interest(now)
            elif args.command == "stop":
                runtime.request_stop(now)
            elif args.command in {"poll", "watch"}:
                if args.command == "watch" and args.interval < 2:
                    raise ExecutionError("poll_interval_too_short")
                stopped = threading.Event()
                while True:
                    runtime.tick(int(time.time() * 1000))
                    status = runtime.status()
                    flow = status["flow"]
                    if (
                        args.command == "poll"
                        or not flow
                        or flow["status"] in {"completed", "expired", "failed", "needs_review"}
                    ):
                        break
                    stopped.wait(args.interval)
        print(json.dumps(runtime.status(), ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        # Any in-flight call already has a durable 'sending' intent. Next poll
        # queries it before considering a new submission.
        return 130
    except ExecutionError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    except Exception:
        # Library calls expose fixed diagnostic codes; a CLI must also avoid raw
        # configuration validation errors containing secret field inputs.
        print(
            json.dumps(
                {
                    "error": "execution_command_failed",
                    "details": "Inspect the journal; no write is automatically retried.",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
