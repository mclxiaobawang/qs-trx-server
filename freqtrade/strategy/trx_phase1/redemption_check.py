"""Operator-run minimum redemption checks, isolated from every strategy ledger.

prepare/verify are GET-only. submit requires the human operator's --execute flag.
An attempted request is durable before HTTP and is never sent again.
"""

import argparse
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from freqtrade.strategy.trx_phase1.account import load_settings
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.live_broker import (
    DAY,
    ClassicBroker,
    ClassicTransport,
    ExecutionError,
    WriteNotSent,
    WriteRejected,
    identifier,
)
from freqtrade.strategy.trx_phase1.live_execution import ProductRule, allocation_lock, digest
from freqtrade.strategy.trx_phase1.request_limits import RequestLimiter


class RedemptionCheck:
    def __init__(self, root, broker, clock=None):
        self.root, self.broker = Path(root), broker
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.folder = self.root / "execution_checks" / "minimum-redemption-2026-10-17"

    @contextmanager
    def locked(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.folder.chmod(0o700)
        fd = os.open(self.folder / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        with allocation_lock(self.root), os.fdopen(fd, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def path(self, coin):
        if coin not in {"USDGO", "TRX"}:
            raise ExecutionError("unsupported_redemption_test_coin")
        return self.folder / (coin + ".json")

    def load(self, coin):
        path = self.path(coin)
        return json.loads(path.read_text()) if path.exists() else None

    def save(self, plan):
        path = self.path(plan["coin"])
        temporary = path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as out:
            json.dump(plan, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        temporary.replace(path)
        directory = os.open(self.folder, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def rule(self, coin):
        value = json.loads((self.root / "live-product-rules.local.json").read_text())
        matches = [p for p in value["products"] if p["coin"] == coin]
        if len(matches) != 1:
            raise ExecutionError("selected_product_rule_unavailable")
        rule = ProductRule.model_validate(matches[0])
        if rule.redemption_minimum is None:
            raise ExecutionError("redemption_terms_unverified")
        return rule

    def observe(self, rule, now):
        if (self.root / "live-strategy-control.json").exists():
            raise ExecutionError("standalone_test_requires_unallocated_account")
        chain = self.root / "execution_checks/minimum-full-chain-2026-10-17/control.json"
        if chain.exists() and not json.loads(chain.read_text()).get("released_at"):
            raise ExecutionError("finish_acceptance_before_another_test")
        account_id = self.broker.account_id()
        product = self.broker.product(rule.coin, rule.product_id)
        if product["status"] != "in_progress":
            raise ExecutionError("selected_product_unavailable")
        holdings = self.broker.holdings(now - 89 * DAY, now)
        candidates = [
            p
            for p in holdings.values()
            if p["coin"] == rule.coin
            and p["product_id"] == rule.product_id
            and money(p["amount"]) >= rule.redemption_minimum
        ]
        if len(candidates) != 1:
            raise ExecutionError("test_holding_missing_or_ambiguous")
        return account_id, candidates[0], self.broker.balances()[rule.coin]

    def prepare(self, coin):
        with self.locked():
            existing = self.load(coin)
            if existing and existing.get("attempted_at"):
                return existing
            rule, now = self.rule(coin), self.clock()
            account_id, holding, balance = self.observe(rule, now)
            # History is auxiliary. Receipt ownership comes from the accepted
            # business order and its exact bill, never a history-row ID guess.
            history = self.history_link(coin, now - 300000, now, None, rule.redemption_minimum)
            self.broker.bills(coin, now - 300000, now)
            plan = {
                "schema": 1,
                "coin": coin,
                "account_id": account_id,
                "rules_sha256": digest(rule.model_dump(mode="json")),
                "prepared_at": now,
                "status": "ready_for_operator_submission",
                "attempted_at": None,
                "exchange_id": None,
                "request": {
                    "productId": rule.product_id,
                    "periodType": "flexible",
                    "amount": format(rule.redemption_minimum, "f"),
                    "orderId": holding["order_id"],
                },
                "before_holding": holding,
                "before_balance": balance,
                "financial_write_attempts": 0,
                "read_checks": {
                    "account_identity": True,
                    "holdings": True,
                    "spendable_balance": True,
                    "product_terms": True,
                    "savings_records": history["status"] != "unavailable",
                    "spot_bills": True,
                },
                "history_error": history.get("error"),
                "error": None,
            }
            self.save(plan)
            return plan

    def submit(self, coin, *, execute=False):  # noqa: C901 - atomic single-request boundary
        if not execute or not self.broker.writes_enabled:
            raise ExecutionError("operator_execute_flag_required")
        with self.locked():
            plan, now = self.load(coin), self.clock()
            if not plan or plan.get("attempted_at"):
                raise ExecutionError("request_already_attempted_or_not_prepared")
            if not 0 <= now - plan["prepared_at"] <= 1800000:
                raise ExecutionError("refresh_read_only_test_preparation")
            for other in ("USDGO", "TRX"):
                previous = self.load(other) or {}
                attempted = previous.get("attempted_at")
                finished = previous.get("request_finished_at") or (
                    attempted + 20000 if attempted else 0
                )
                if attempted is not None and now - finished < 61000:
                    raise ExecutionError("redemption_test_cooldown_61_seconds")
                if attempted is not None and previous["status"] != "verified":
                    raise ExecutionError("verify_previous_redemption_before_next_coin")
            rule = self.rule(coin)
            if digest(rule.model_dump(mode="json")) != plan["rules_sha256"]:
                raise ExecutionError("test_product_rules_changed")
            account_id, holding, balance = self.observe(rule, now)
            if (
                account_id != plan["account_id"]
                or holding["order_id"] != plan["request"]["orderId"]
            ):
                raise ExecutionError("test_account_or_holding_changed")
            expected = {
                "productId": rule.product_id,
                "periodType": "flexible",
                "amount": format(rule.redemption_minimum, "f"),
                "orderId": holding["order_id"],
            }
            if plan["request"] != expected:
                raise ExecutionError("test_request_changed")
            if hasattr(self.broker, "reserve_submission"):
                self.broker.reserve_submission({"kind": "redeem"}, self.clock() + 10000)
            plan.update(
                attempted_at=self.clock(),
                before_holding=holding,
                before_balance=balance,
                status="sending",
                financial_write_attempts=1,
            )
            self.save(plan)  # A crash or timeout after this point forbids a second POST.
            try:
                result = self.broker.submit(
                    {
                        "kind": "redeem",
                        "id": "minimum-" + coin,
                        "created_at": now,
                        "request": plan["request"],
                    }
                )
                plan.update(exchange_id=identifier(result["orderId"]), status="accepted_unverified")
            except WriteNotSent:
                plan.update(
                    status="prepared",
                    attempted_at=None,
                    financial_write_attempts=0,
                    error="request_rate_wait",
                )
            except WriteRejected as exc:
                plan.update(status="rejected_requires_review", error=str(exc))
            except ExecutionError as exc:
                plan.update(status="outcome_unknown_requires_review", error=str(exc))
            except Exception:
                plan.update(
                    status="outcome_unknown_requires_review", error="redemption_response_unknown"
                )
            plan["request_finished_at"] = self.clock()
            self.save(plan)
            return plan

    def history_link(self, coin, start, end, exchange_id, amount):
        link = {"accepted_exchange_order_id": exchange_id, "candidate_records": []}
        try:
            rows = self.broker.savings_records(coin, start, end, order_type="redeem")
            for row in rows:
                if (
                    row.get("orderType") == "redeem"
                    and row.get("coinName") == coin
                    and row.get("settleCoinName") == coin
                    and row.get("productType") == "flexible"
                    and money(row["amount"]) == amount
                    and start <= int(row["ts"]) <= end
                ):
                    link["candidate_records"].append(
                        {
                            "history_record_id": identifier(row["orderId"]),
                            "coin": coin,
                            "settlement_coin": coin,
                            "kind": "redeem",
                            "amount": str(money(row["amount"])),
                            "timestamp": int(row["ts"]),
                            "direct_order_id_match": str(row["orderId"]) == exchange_id,
                        }
                    )
            direct = [r for r in link["candidate_records"] if r["direct_order_id_match"]]
            link["status"] = (
                "verified"
                if len(direct) == 1
                else "ambiguous_records"
                if len(link["candidate_records"]) > 1
                else "record_id_unlinked"
                if link["candidate_records"]
                else "record_not_observed"
            )
        except Exception:
            link.update(
                status="unavailable", candidate_records=[], error="history_read_unavailable"
            )
        return link

    def verify(self, coin, exchange_id=None):
        with self.locked():
            plan, now = self.load(coin), self.clock()
            if not plan or not plan.get("attempted_at"):
                raise ExecutionError("no_redemption_test_submission")
            if self.broker.account_id() != plan["account_id"]:
                raise ExecutionError("account_binding_mismatch")
            if exchange_id:
                exchange_id = identifier(exchange_id)
                if plan["exchange_id"] not in (None, exchange_id):
                    raise ExecutionError("redemption_order_identity_conflict")
                plan["exchange_id"] = exchange_id
                plan["operator_bound_exchange_id"] = True
                self.save(plan)
            if not plan["exchange_id"]:
                return plan  # An ambiguous POST cannot be identified from amount/time alone.
            oid, amount = plan["exchange_id"], money(plan["request"]["amount"])
            start = plan["attempted_at"] - 1000
            result = self.broker.result("redeem", oid)
            bills = [
                b
                for b in self.broker.bills(coin, start, now)
                if str(b.get("bizOrderId")) == oid
                and b.get("coin") == coin
                and b.get("groupType") == "financial"
                and start <= int(b["cTime"]) <= now
            ]
            holdings = self.broker.holdings(plan["prepared_at"] - 89 * DAY, now)
            remaining = money(holdings.get(plan["request"]["orderId"], {}).get("amount", 0))
            wallet = self.broker.balances()[coin]
            checks = {
                "exchange_result": result.get("result") == "success",
                "arrival_bill": bool(bills)
                and sum((money(b["size"]) for b in bills), money(0)) == amount
                and all(money(b["fees"]) == 0 for b in bills),
                "holding_reduction": money(plan["before_holding"]["amount"]) - remaining == amount,
                "spendable_arrival": money(wallet["available"])
                - money(plan["before_balance"]["available"])
                == amount
                and all(
                    money(wallet[k]) == money(plan["before_balance"][k])
                    for k in ("frozen", "locked", "restricted")
                ),
            }
            status = (
                "verified"
                if all(checks.values())
                else "failed_requires_review"
                if result.get("result") == "fail"
                else "pending_or_conflicting_evidence"
            )
            plan.update(
                checked_at=now,
                verification_version=2,
                verification_basis="business_order_bill_holding_spendable_balance",
                checks=checks,
                arrival_status=status,
                history_link=self.history_link(coin, start, now, oid, amount),
                after_balance=wallet,
                after_holding_amount=str(remaining),
                status=status,
            )
            self.save(plan)
            return plan


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Operator-controlled minimum redemption acceptance"
    )
    parser.add_argument("--root", type=Path, default=Path("user_data/trx_phase1"))
    parser.add_argument("command", choices=("prepare", "submit", "verify"))
    parser.add_argument("--coin", choices=("USDGO", "TRX"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--exchange-order-id")
    args = parser.parse_args(argv)
    transport = None
    try:
        if args.command == "submit" and (not args.coin or not args.execute):
            raise ExecutionError("operator_coin_and_execute_flag_required")
        settings, _, complete = load_settings(args.root / "bitget-account.local.json")
        if not complete or not settings.enabled:
            raise ExecutionError("enabled_credentials_required")
        transport = ClassicTransport(
            settings,
            writes_enabled=args.command == "submit" and args.execute,
            limiter=RequestLimiter(args.root / "request-limits.sqlite"),
        )
        transport.sync_clock()
        check = RedemptionCheck(args.root, ClassicBroker(transport))
        values = []
        for coin in (args.coin,) if args.coin else ("USDGO", "TRX"):
            if args.command == "prepare":
                value = check.prepare(coin)
            elif args.command == "submit":
                value = check.submit(coin, execute=args.execute)
                if value["exchange_id"]:
                    value = check.verify(coin)
            else:
                value = check.verify(coin, args.exchange_order_id)
            values.append(value)
        print(json.dumps(values, ensure_ascii=False, indent=2))
        return 0
    except ExecutionError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    except Exception:
        print(json.dumps({"error": "redemption_check_failed_read_local_journal"}))
        return 1
    finally:
        if transport:
            transport.session.close()


if __name__ == "__main__":
    raise SystemExit(main())
