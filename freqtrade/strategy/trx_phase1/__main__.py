import argparse
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from freqtrade.strategy.trx_phase1.config import MECHANISMS, ResearchConfig
from freqtrade.strategy.trx_phase1.replay import replay


def main():
    parser = argparse.ArgumentParser(description="TRX lot/NAV research on local 1m spot candles")
    parser.add_argument(
        "--data", type=Path, default=Path("user_data/data/bitget/TRX_USDT-1m.feather")
    )
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--variant", choices=list(MECHANISMS), default="dynamic_period")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = (
        ResearchConfig.from_saved(json.loads(args.parameters.read_text()))
        if args.parameters
        else ResearchConfig(variant=args.variant, earn_model="simplified_hourly")
    )
    result = replay(pd.read_feather(args.data), config)
    run_id = str(uuid4())
    result["run_id"] = run_id
    output = args.output or Path("user_data/trx_phase1/results") / f"{run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, allow_nan=False))
    print(json.dumps({"file": str(output.resolve()), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
