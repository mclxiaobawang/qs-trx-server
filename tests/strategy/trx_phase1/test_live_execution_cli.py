import json

from freqtrade.strategy.trx_phase1 import live_execution_cli as cli
from tests.strategy.trx_phase1.test_live_execution import ACCOUNT, NOW, ExchangeFixture, trade


def test_operator_cli_is_inert_until_execute_and_recovers_read_only(tmp_path, mocker, capsys):
    credentials = tmp_path / "credentials.json"
    credentials.write_text(
        json.dumps(
            {
                "account_type": "classic",
                "api_key": "fixture-key",
                "secret_key": "fixture-secret",
                "passphrase": "fixture-passphrase",
            }
        )
    )
    credentials.chmod(0o600)
    products = tmp_path / "products.json"
    products.write_text("[]")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"label": "CLI synthetic workflow", "steps": [trade()]}))
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    mocker.patch.object(cli.ClassicTransport, "sync_clock")
    clock = mocker.patch(
        "freqtrade.strategy.trx_phase1.live_execution_cli.time.time", return_value=NOW / 1000
    )

    def factory(transport):
        broker.writes_enabled = transport.writes_enabled
        return broker

    mocker.patch.object(cli, "ClassicBroker", side_effect=factory)
    prefix = ["--root", str(tmp_path), "--credentials", str(credentials)]
    assert cli.main(prefix + ["init", "--capital", "1000.4", "--products", str(products)]) == 0
    assert cli.main(prefix + ["queue", "--account-id", ACCOUNT, "--plan", str(plan)]) == 0
    assert cli.main(prefix + ["poll", "--account-id", ACCOUNT]) == 0
    assert cli.main(prefix + ["poll", "--account-id", ACCOUNT]) == 0
    assert broker.state()["writes"] == []
    assert cli.main(prefix + ["poll", "--account-id", ACCOUNT, "--execute"]) == 0
    assert len(broker.state()["writes"]) == 1
    # Submission adds real preflight duration to its timestamp; recovery is later.
    clock.return_value = (NOW + 10000) / 1000
    assert cli.main(prefix + ["poll", "--account-id", ACCOUNT]) == 0
    capsys.readouterr()
    assert cli.main(prefix + ["status", "--account-id", ACCOUNT]) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["flow"]["status"] == "completed"
    assert state["operation"] is None
    assert "fixture-secret" not in json.dumps(state)


def test_invalid_local_credentials_never_print_secret_values(tmp_path, capsys):
    credentials = tmp_path / "credentials.json"
    credentials.write_text(
        json.dumps(
            {
                "account_type": "classic",
                "api_key": "fixture-key",
                "secret_key": "fixture-secret",
                "passphrase": "fixture-passphrase",
                "unexpected": "fixture-private-value",
            }
        )
    )
    credentials.chmod(0o600)
    assert (
        cli.main(
            [
                "--credentials",
                str(credentials),
                "init",
                "--capital",
                "1",
                "--products",
                str(tmp_path / "unused.json"),
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "execution_command_failed" in output
    assert "fixture-private-value" not in output
    assert "fixture-secret" not in output
