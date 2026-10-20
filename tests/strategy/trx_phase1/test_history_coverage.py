import pandas as pd
import pytest

from freqtrade.strategy.trx_phase1.history_coverage import (
    coverage,
    frame_from_rows,
    select_continuous_interval,
)
from tests.strategy.trx_phase1.test_diagnostics import bars


def test_inventory_gaps_and_select_only_the_continuous_component_into_control():
    frame = bars(1500)
    original_start, end = frame.date.iloc[1000], frame.date.iloc[-1]
    incomplete = frame.drop(index=[200, 201, 700])
    report = coverage(incomplete)
    assert report["missing_minutes"] == 3
    assert [x["missing_minutes"] for x in report["gaps"]] == [2, 1]
    selected = select_continuous_interval(report, original_start, end)
    assert selected["start_utc"] == str(frame.date.iloc[701])
    assert selected["rows"] == 799
    assert report["synthetic_minutes"] == 0
    with pytest.raises(ValueError, match="No earlier continuous"):
        select_continuous_interval(report, frame.date.iloc[600], end)


def test_duplicate_invalid_and_exchange_rows_are_not_silently_repaired():
    frame = bars(10)
    with pytest.raises(ValueError, match="duplicate"):
        coverage(pd.concat([frame, frame.iloc[:1]]))
    frame.loc[0, "high"] = 0
    with pytest.raises(ValueError, match="invalid actual"):
        coverage(frame)
    rows = [[60000, "1", "2", "0.5", "1.2", "0", "0", "0"]]
    report = coverage(frame_from_rows(rows))
    assert report["rows"] == 1 and report["synthetic_minutes"] == 0
    assert coverage(frame_from_rows([]))["rows"] == 0
