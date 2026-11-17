"""Sensitivity tests, not labels: console launches scaled to GTA's release day.

H1 preserves each donor's ratios. H2 applies the console-only OLS slope of
mean log(viewers/hour1) at forecast hours 15..23 against log(hour1), retaining
each donor's residual shape. Its log adjustment ramps linearly from zero at
hour 1 to the full adjustment at hour 15, and stays constant thereafter.
The 12 console launches include the four practices and the early-access-marked
Death Stranding 2; no donor is used to forecast itself. Results include both
missing-platform and final-shaped platform-snapshot inputs.
"""

import argparse
import importlib.util
import json
import math
import statistics
import tempfile
from datetime import timedelta
from pathlib import Path

import backtest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT.parent / "evidence"
TARGET_DAY = "2026-11-19T00:00:00Z"


def evidence_path(path):
    path = path.resolve()
    if not path.is_relative_to(EVIDENCE.resolve()):
        raise ValueError("experiment outputs must be under ../evidence/")
    return path


def load_model(path):
    spec = importlib.util.spec_from_file_location("stress_entry", path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def load_pool(kit, answers):
    practices = [json.loads(p.read_text()) for p in sorted((kit / "input").glob("p*.in"))]
    return backtest.final_shape(practices, practices[0]["comparables"],
                                backtest.load_answers(answers))["comparables"]


def console_launches(pool):
    return [r for r in pool if not backtest.is_target_history(r) and backtest.console_only(r)
            and backtest.synthesize(r, pool, "probe") is not None]


def console_trend(pool):
    points = []
    for launch in console_launches(pool):
        case, truth = backtest.synthesize(launch, pool, "trend")
        p = case["known"][0]["viewers"]
        y = statistics.mean(math.log(truth[h] / p) for h in range(15, 24))
        points.append({"name": launch["name"], "logp": math.log(p), "late_log_ratio": y})
    xm = statistics.mean(r["logp"] for r in points)
    ym = statistics.mean(r["late_log_ratio"] for r in points)
    ss = sum((r["logp"] - xm) ** 2 for r in points)
    slope = sum((r["logp"] - xm) * (r["late_log_ratio"] - ym) for r in points) / ss
    intercept = ym - slope * xm
    se = math.sqrt(sum((r["late_log_ratio"] - intercept - slope * r["logp"]) ** 2
                       for r in points) / (len(points) - 2) / ss)
    return {"n": len(points), "slope": slope, "standard_error": se,
            "intercept": intercept, "points": points}


def add_platform(case, p, total):
    """Top twenty = GTA plus a background UTC cycle, independent of GTA's eve.

    `total` is an assumed launch-hour platform audience, not a historical label.
    Other categories keep their own daily shape as the target game surges.
    """
    start = backtest.instant(case["day_start"])
    weights = [1 / (i + 1) ** .8 for i in range(19)]
    norm = sum(weights)
    case["categories_hourly"] = []
    for row in case["history_hourly"]:
        hour = int((backtest.instant(row["from"]) - start).total_seconds() // 3600)
        cycle = (.82 + .18 * math.cos(2 * math.pi * (hour - 20) / 24)) / .91
        background = max(500000, .85 * total - p) * cycle
        top = [{"name": case["game"]["name"], "viewers": row["viewers"],
                "channels": row.get("channels")}]
        top.extend({"name": f"Category {i + 1}", "viewers": round(background * w / norm),
                    "channels": 3000 // (i + 1)} for i, w in enumerate(weights))
        case["categories_hourly"].append({"from": row["from"],
                                          "top": sorted(top, key=lambda r: -r["viewers"])})


def scaled_case(launch, pool, p, total=None):
    case, truth = backtest.synthesize(launch, pool, "h00")
    original_p = case["known"][0]["viewers"]
    scale = p / original_p
    shift = backtest.instant(TARGET_DAY) - backtest.instant(case["day_start"])
    case["game"] = {"name": "Grand Theft Auto VI", "twitch_id": "1867279146"}
    case["notes"] = "PS5, Xbox Series; rolling midnight local. US east 05:00Z, west 08:00Z."
    for key in ("day_start", "checkpoint"):
        case[key] = backtest.iso(backtest.instant(case[key]) + shift)
    for row in case["known"] + case["forecast"]:
        for key in ("from", "until"):
            row[key] = backtest.iso(backtest.instant(row[key]) + shift)
        if "viewers" in row:
            row["viewers"] = p
            row["reads"] = 60
        if row.get("channels") is not None:
            row["channels"] *= scale
    cutoff = backtest.instant(TARGET_DAY) - timedelta(hours=48)
    history = []
    for row in case["history_hourly"]:
        at = backtest.instant(row["from"]) + shift
        if at < cutoff:
            continue
        row["from"] = backtest.iso(at)
        row["viewers"] *= scale
        if row.get("channels") is not None:
            row["channels"] *= scale
        row["reads"] = 60
        history.append(row)
    case["history_hourly"] = history
    case["history"] = [{"at": backtest.iso(backtest.instant(r["from"]) + timedelta(minutes=m)),
                        "viewers": r["viewers"], "channels": r.get("channels")}
                       for r in history for m in range(60)]
    if total is not None:
        add_platform(case, p, total)
    return case, {h: y * scale for h, y in truth.items()}, scale


def evaluate(model, pool, scales=(1000000, 2000000), platform=True, subprocess=False):
    trend = console_trend(pool)
    rows = []
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gta-scale-", dir=EVIDENCE) as directory:
        temp = Path(directory)
        for p in scales:
            for launch in console_launches(pool):
                # Assumptions deliberately shared between before/after runs.
                total = 4000000 if p == 1000000 else 5000000
                case, h1, scale = scaled_case(launch, pool, p, total if platform else None)
                assert all(r["name"] != launch["name"] for r in case["comparables"])
                result = model.forecast_case(case)
                pred = backtest.validate(json.dumps(result).encode())
                capacity_active = model.platform_context(case, p, model.comparable_records(case)) is not None
                assert capacity_active == platform, "platform stress silently disabled capacity"
                if subprocess:
                    _, elapsed, raw = backtest.run_entry(case, temp, Path(model.__file__))
                    _, _, raw2 = backtest.run_entry(case, temp, Path(model.__file__))
                    assert raw == raw2 and json.loads(raw) == result, "fallback or nondeterminism"
                else:
                    elapsed = None
                h2 = {h: y * math.exp(trend["slope"] * math.log(scale) * min(1, (h - 1) / 14))
                      for h, y in h1.items()}
                rows.append({"name": launch["name"], "hour1": p, "platform": platform,
                             "capacity_active": capacity_active,
                             "H1": backtest.score(pred, h1), "H2": backtest.score(pred, h2),
                             "seconds": elapsed})
    return {"trend": trend, "rows": rows, "summary": [
        {"hour1": p, "platform": platform,
         **{hyp: backtest.aggregate([r[hyp] for r in rows if r["hour1"] == p]) for hyp in ("H1", "H2")}}
        for p in scales]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=ROOT.parent / "gta6-twitch-ccu-forecast-01-kit")
    parser.add_argument("--answers", type=Path, default=EVIDENCE / "gta-practice-answers.json")
    parser.add_argument("--entry", type=Path, default=ROOT / "main.py")
    parser.add_argument("--report", type=Path, default=EVIDENCE / "scale-stress.json")
    parser.add_argument("--subprocess", action="store_true", help="also verify real entry, twice per case")
    args = parser.parse_args()
    output = evidence_path(args.report)
    model = load_model(args.entry)
    pool = load_pool(args.kit, args.answers)
    reports = [evaluate(model, pool, platform=platform, subprocess=args.subprocess) for platform in (False, True)]
    trend = reports[0]["trend"]
    print(f"Console late log-ratio slope: {trend['slope']:.6f} +/- {trend['standard_error']:.6f} SE; n={trend['n']}")
    print("Synthetic truths; assumed platform 4M / 5M at hour 1 = 1M / 2M.")
    print("platform hour1 launch H1_loss H1_90% H2_loss H2_90%")
    for report in reports:
        for r in report["rows"] + [dict(r, name="MEAN") for r in report["summary"]]:
            print(f"{str(r['platform']):8} {r['hour1']:7} {r['name']:38} "
                  f"{r['H1']['loss']:.6f} {r['H1']['coverage']:.3f} "
                  f"{r['H2']['loss']:.6f} {r['H2']['coverage']:.3f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
