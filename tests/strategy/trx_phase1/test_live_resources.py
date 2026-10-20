import json
from types import SimpleNamespace

from freqtrade.strategy.trx_phase1 import live_resources


def test_sampler_resolves_current_container_cgroup_and_only_inspects(tmp_path, monkeypatch):
    container = {
        "Id": "first",
        "State": {"Pid": 10, "Running": True, "StartedAt": "first"},
        "RestartCount": 0,
    }
    commands = []

    def inspect(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(stdout=json.dumps([container]))

    monkeypatch.setattr(live_resources.subprocess, "run", inspect)
    proc, cgroups = tmp_path / "proc", tmp_path / "cgroup"
    for pid, group, used in ((10, "old", 800), (20, "new", 200)):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "cgroup").write_text("0::/" + group)
        (cgroups / group).mkdir(parents=True)
        (cgroups / group / "memory.current").write_text(str(used))
        (cgroups / group / "memory.max").write_text("1000")
    assert live_resources.collect(proc, cgroups)["current"] == 800
    container.update(Id="second", RestartCount=1)
    container["State"].update(Pid=20, StartedAt="second")
    sample = live_resources.collect(proc, cgroups)
    assert sample["current"] == 200 and sample["container_id"] == "second"
    assert commands == [["docker", "inspect", "trx-phase1-live"]] * 2
    target = tmp_path / "sample.json"
    live_resources.save(target, sample)
    assert json.loads(target.read_text()) == sample
    assert target.stat().st_mode & 0o777 == 0o600
    container["State"]["Running"] = False
    assert live_resources.collect(proc, cgroups)["error"] == "ValueError"
