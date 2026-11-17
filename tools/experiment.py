"""Small, declared ablations. Select by held-out comparables, then check practice.

These development runs call the same model in-process; backtest.py is the
authoritative subprocess, validity, time-limit, and determinism check.
"""

import importlib.util
import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import backtest

spec = importlib.util.spec_from_file_location("entry", ROOT / "main.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def evaluate(practices, launches, answers):
    prows, loo, like = [], [], []
    for case in practices:
        prediction = backtest.validate(json.dumps(entry.forecast_case(case)).encode())
        prows.append(backtest.score(prediction, answers[case["id"]])["loss"])
    for i, launch in enumerate(launches):
        if backtest.is_target_history(launch):
            continue
        pair = backtest.synthesize(launch, launches, f"loo{i:02}")
        if pair is None:
            continue
        case, truth = pair
        prediction = backtest.validate(json.dumps(entry.forecast_case(case)).encode())
        loss = backtest.score(prediction, truth)["loss"]
        loo.append(loss)
        if backtest.gta_like(launch):
            like.append(loss)
    return prows, sum(loo) / len(loo), sum(like) / len(like)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural", action="store_true")
    parser.add_argument("--activity", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    kit = ROOT.parent / "gta6-twitch-ccu-forecast-01-kit"
    practices = [json.loads(p.read_text()) for p in sorted((kit / "input").glob("p*.in"))]
    answers = backtest.load_answers(Path("/tmp/gta-practice-answers.json"))
    launches = practices[0]["comparables"]
    variants = [
        ("base pooled wide", "base", False, 3.0, .50, 1.00),
        ("channels pooled", "channels", False, 3.0, .25, .85),
        ("base active", "base", True, 3.0, .25, .85),
        ("channels active", "channels", True, 3.0, .25, .85),
        ("base ridge1", "base", False, 1.0, .25, .85),
        ("channels ridge1", "channels", False, 1.0, .25, .85),
        ("base ridge10", "base", False, 10.0, .25, .85),
        ("channels ridge10", "channels", False, 10.0, .25, .85),
        ("base full spread", "base", False, 3.0, .25, 1.00),
        ("base local residual", "base", False, 3.0, .00, .85),
        ("base pooled residual", "base", False, 3.0, .75, .85),
    ]
    if args.structural:
        variants = [
            ("full cap13", "channels", False, 3.0, .25, .85, 13.0),
            ("full cap12.5", "channels", False, 3.0, .25, .85, 12.5),
            ("full cap12", "channels", False, 3.0, .25, .85, 12.0),
            ("active cap13", "channels", True, 3.0, .25, .85, 13.0),
            ("active cap12.5", "channels", True, 3.0, .25, .85, 12.5),
            ("active cap12", "channels", True, 3.0, .25, .85, 12.0),
            ("active base", "base", True, 3.0, .25, .85, 12.5),
            ("active ridge1", "channels", True, 1.0, .25, .85, 12.5),
            ("active ridge10", "channels", True, 10.0, .25, .85, 12.5),
            ("active full spread", "channels", True, 3.0, .25, 1.0, 12.5),
        ]
    if args.activity:
        variants = [
            ("activity ridge1", "activity", False, 1.0, .25, .85, float("inf")),
            ("activity ridge3", "activity", False, 3.0, .25, .85, float("inf")),
            ("activity ridge10", "activity", False, 10.0, .25, .85, float("inf")),
            ("activity cap13", "activity", False, 3.0, .25, .85, 13.0),
            ("activity full spread", "activity", False, 3.0, .25, 1.0, float("inf")),
        ]
    if args.sparse:
        variants = [(f"sparse s{spread} j{int(jitter)}", "activity", False, 3.0, .25, .85, 13.0, spread, jitter)
                    for spread, jitter in ((.25, False), (.45, False), (.25, True), (.45, True), (.65, True))]
    print("variant                    p00     p01     p02     p03    Pmean  LOOall LOOlike", flush=True)
    for variant in variants:
        name, feature, regime, ridge, pool, scale = variant[:6]
        entry.LOGP_CAP = variant[6] if len(variant) > 6 else float("inf")
        entry.SPARSE_SPREAD = variant[7] if len(variant) > 7 else .25
        entry.SPARSE_JITTER = variant[8] if len(variant) > 8 else False
        entry.FEATURE_SET, entry.REGIME, entry.RIDGE = feature, regime, ridge
        entry.RESIDUAL_POOL, entry.RESIDUAL_SCALE = pool, scale
        started = time.perf_counter()
        prows, all_loss, like_loss = evaluate(practices, launches, answers)
        print(f"{name:26}" + "".join(f" {x:.4f}" for x in prows) +
              f" {sum(prows)/4:.4f} {all_loss:.4f} {like_loss:.4f} ({time.perf_counter()-started:.1f}s)", flush=True)
    prows = [backtest.score(backtest.persistence(c["known"][0]["viewers"]), answers[c["id"]])["loss"]
             for c in practices]
    rows, like = [], []
    for i, launch in enumerate(launches):
        if backtest.is_target_history(launch):
            continue
        pair = backtest.synthesize(launch, launches, f"loo{i:02}")
        if pair:
            case, truth = pair
            row = backtest.score(backtest.persistence(case["known"][0]["viewers"]), truth)
            rows.append(row)
            if backtest.gta_like(launch):
                like.append(row)
    print(f"{'Baseline':26}" + "".join(f" {x:.4f}" for x in prows) +
          f" {sum(prows)/4:.4f} {backtest.aggregate(rows)['loss']:.4f} {backtest.aggregate(like)['loss']:.4f}")


if __name__ == "__main__":
    main()
