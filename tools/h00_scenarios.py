"""Replay realistic release-day inputs; write artifacts outside the repository.

python3 tools/h00_scenarios.py --output ../evidence/h00-scenarios
Values printed in the tables are thousands of concurrent viewers, never labels.
"""

import argparse
import copy
import json
import math
import tempfile
from datetime import timedelta
from pathlib import Path

import backtest
from scale_stress import add_platform, evidence_path, load_model


ROOT = Path(__file__).resolve().parents[1]


DONORS = ("God of War Ragnarök", "Marvel's Spider-Man 2", "Red Dead Redemption 2", "The Last of Us Part II")


def scenario(template, p, total, donor="God of War Ragnarök"):
    case = copy.deepcopy(template)
    case["id"] = "h00"
    case["history"] = []
    case["history_hourly"] = []
    case["categories_hourly"] = []
    case["notes"] = ("PS5 and Xbox Series; rolling midnight local. NZ 2026-11-18T11:00Z; "
                     "CET 23:00Z, UK 00:00Z, US east 05:00Z, west 08:00Z. "
                     "US streamers can use the Xbox NZ region; US prime time is 23:00Z-04:00Z.")
    start = backtest.instant(case["day_start"])
    launch = next(r for r in template["comparables"] if r["name"] == donor)
    original_start = backtest.instant(launch["day_start"])
    shape = {int((backtest.instant(r["from"]) - original_start).total_seconds() // 3600): r["viewers"]
             for r in launch["hourly"]}
    peak = max(range(-24, 0), key=lambda h: shape[h])
    # TLOU2's late 22Z peak is moved to 20Z, stretching its decline to 00Z.
    # The other three retain their observed 17Z, 18Z and 19Z peaks exactly.
    aligned_peak = min(-4, max(-7, peak))
    def eve(h):
        source = h + peak - aligned_peak if h <= aligned_peak else h * peak / aligned_peak
        lo = math.floor(source)
        fraction = source - lo
        return ((1 - fraction) * shape[lo] + fraction * shape.get(lo + 1, shape[lo])) / shape[0]

    for h in range(-48, 1):
        at = start + timedelta(hours=h)
        quiet = 1900 + 650 * math.cos(2 * math.pi * (h - 19) / 24)
        mean = round(quiet if h < -24 else p * eve(h))
        channels = 160 if h < -24 else max(160, round(mean / 100))
        for minute in range(60):
            case["history"].append({"at": backtest.iso(at + timedelta(minutes=minute)),
                                    "viewers": mean + round(mean * .02 * (2 * minute - 59) / 60),
                                    "channels": channels})
        case["history_hourly"].append({"from": backtest.iso(at), "viewers": mean,
                                       "channels": channels, "reads": 60})
    add_platform(case, p, total)
    case["known"] = [{"hour": 1, "from": backtest.iso(start),
                      "until": backtest.iso(start + timedelta(hours=1)),
                      "viewers": p, "channels": round(p / 100), "reads": 60}]
    case["streams"] = [{"login": f"stream{i:02}", "viewers": round(.14 * p / (i + 1)),
                        "language": ("en", "de", "es", "ja", "pt")[i % 5],
                        "started_at": backtest.iso(start - timedelta(hours=13 - i % 8))}
                       for i in range(20)]
    return case


def check_scenarios(cases, entry):
    """Check release-day behavior directly, so a valid fallback cannot hide bugs."""
    model = load_model(entry)
    for case in cases:
        p = case["known"][0]["viewers"]
        forecast = model.forecast_case(case)
        assert model.valid_forecast(forecast)
        median = {r["hour"]: r["q"][2] for r in forecast["hours"]}
        # Hour 1 starts thirteen hours after NZ unlock, never at the beginning
        # of the quiet period. Check timestamps and observable forecast shape.
        start = backtest.instant(case["day_start"])
        assert case["checkpoint"] == "2026-11-19T01:00:00Z"
        assert case["history"][0]["at"] == "2026-11-17T00:00:00Z"
        quiet = [r["viewers"] for r in case["history"]
                 if backtest.instant(r["at"]) < start - timedelta(hours=24)]
        assert 1000 < min(quiet) <= max(quiet) < 3000
        eve = case["history_hourly"][-25:-1]
        peak = max(eve, key=lambda r: r["viewers"])
        assert 17 <= backtest.instant(peak["from"]).hour <= 20
        assert peak["viewers"] > eve[-1]["viewers"] > p
        assert min(median.values()) > p / 4
        assert max(median[6], median[7]) > median[4], "missing US unlock response"
        assert median[18] > median[11], "missing daytime recovery"
        assert median[24] < median[21], "missing late-evening decline"
        minute = copy.deepcopy(case)
        minute["history_hourly"] = None
        assert model.forecast_case(minute) == forecast
        ignored = copy.deepcopy(case)
        ignored["streams"] = None
        assert model.forecast_case(ignored) == forecast
        ignored["categories_hourly"].append({"from": case["checkpoint"], "top": [
            {"name": "leaked future", "viewers": 1e9}] * 20})
        ignored["history"].append({"at": case["checkpoint"], "viewers": 1e9, "channels": 1})
        ignored["history_hourly"].append({"from": case["checkpoint"], "viewers": 1e9, "channels": 1})
        assert model.forecast_case(ignored) == forecast, "checkpoint leak"
        no_platform = copy.deepcopy(case)
        no_platform["categories_hourly"] = None
        unadjusted = model.forecast_case(no_platform)
        if p == 300000:
            assert forecast == unadjusted, "adjustment touched an observed launch size"
        else:
            assert model.platform_context(case, p, model.comparable_records(case)) is not None
            # A low-capacity replay must visibly bind, so merely entering the
            # code path or returning a persistence fallback cannot pass.
            constrained = copy.deepcopy(case)
            for row in constrained["categories_hourly"]:
                for c in row["top"]:
                    if c["name"] != case["game"]["name"]:
                        c["viewers"] *= .1
            limited = model.forecast_case(constrained)
            assert any(a["q"][4] < b["q"][4] for a, b in zip(limited["hours"], unadjusted["hours"])), "capacity ceiling never bound"
            assert all(a["q"][:3] == b["q"][:3] for a, b in zip(limited["hours"], unadjusted["hours"])), "capacity changed median/lower tail"
            for row in constrained["categories_hourly"]:
                if row["from"] < case["day_start"]:
                    for c in row["top"]:
                        if c["name"] == case["game"]["name"]:
                            c["viewers"] *= 100
            assert model.forecast_case(constrained) == limited, "GTA eve leaked into platform cycle"
            # A missing own row and one or two absent category rows must not
            # silently disable an otherwise complete series of snapshots.
            for row in constrained["categories_hourly"]:
                row["top"] = [c for c in row["top"] if c["name"] != case["game"]["name"]]
            assert model.forecast_case(constrained) == limited
            for row in constrained["categories_hourly"]:
                row["top"] = row["top"][:-1]
            assert model.platform_context(constrained, p, model.comparable_records(constrained)) is not None
            incomplete = model.forecast_case(constrained)
            assert model.valid_forecast(incomplete)
            assert any(a["q"][4] < b["q"][4] for a, b in zip(incomplete["hours"], unadjusted["hours"])), "18-row snapshots disabled capacity"
        for bad in ([], [{"from": "bad", "top": None}],
                    [{"from": case["checkpoint"], "top": case["categories_hourly"][-1]["top"]}],
                    [dict(r, top=r["top"][:3]) for r in case["categories_hourly"]]):
            no_platform["categories_hourly"] = bad
            assert model.forecast_case(no_platform) == unadjusted
    # At fixed hour-1 viewers, a smaller measured platform must not raise the
    # upper tail. Stress a near-platform-sized category and a late NZ ramp too.
    high = cases[-1]
    base = model.forecast_case(high)
    smaller = copy.deepcopy(high)
    for row in smaller["categories_hourly"]:
        for c in row["top"]:
            c["viewers"] *= .5
    compressed = model.forecast_case(smaller)
    assert model.valid_forecast(compressed)
    assert all(a["q"][4] <= b["q"][4] for a, b in zip(compressed["hours"], base["hours"]))
    late = copy.deepcopy(high)
    for key, time_key in (("history", "at"), ("history_hourly", "from")):
        for row in late[key]:
            if row[time_key] < late["day_start"]:
                row["viewers"] *= .1
    assert model.valid_forecast(model.forecast_case(late))
    return [model.forecast_case(case) for case in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=ROOT.parent / "gta6-twitch-ccu-forecast-01-kit")
    parser.add_argument("--answers", type=Path, default=ROOT.parent / "evidence/gta-practice-answers.json")
    parser.add_argument("--entry", type=Path, default=ROOT / "main.py")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "evidence/h00-scenarios")
    args = parser.parse_args()
    args.output = evidence_path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    practices = [json.loads(p.read_text()) for p in sorted((args.kit / "input").glob("p*.in"))]
    template = backtest.final_shape(practices, practices[0]["comparables"], backtest.load_answers(args.answers))
    cases = [scenario(template, p, total, donor) for donor in DONORS
             for p, total in ((300000, 3600000), (1000000, 4000000), (2000000, 5000000))]
    direct = check_scenarios(cases, args.entry)
    forecasts = []
    with tempfile.TemporaryDirectory(prefix="gta-h00-", dir=args.output) as directory:
        temp = Path(directory)
        for i, (case, expected) in enumerate(zip(cases, direct)):
            p = case["known"][0]["viewers"]
            pred, _, raw = backtest.run_entry(case, temp, args.entry)
            _, _, raw2 = backtest.run_entry(case, temp, args.entry)
            assert raw == raw2, "nondeterministic scenario"
            assert json.loads(raw) == expected, "entry fell back instead of running the model"
            (args.output / f"h00-{i//3}-{p}.in").write_text(json.dumps(case) + "\n")
            (args.output / f"h00-{i//3}-{p}.out").write_bytes(raw)
            forecasts.append(pred)
    print("Synthetic h00; all values in thousands: median [5%, 95%]. UTC is hour start.")
    print("Checks passed: clock, quiet/NZ history, shape, aggregation, stream independence, "
          "checkpoint cut, scale/share gating, missing categories, and direct/subprocess equality.")
    for i, donor in enumerate(DONORS):
        print(f"\nEve: {donor}" + (" (peak aligned to 20Z)" if i == 3 else ""))
        print("| Hour | UTC | 300k hour 1 | 1M hour 1 | 2M hour 1 |")
        print("|---:|:---:|---:|---:|---:|")
        for h in range(2, 25):
            cells = [f"{pred[h][2]/1000:.0f} [{pred[h][0]/1000:.0f}, {pred[h][4]/1000:.0f}]"
                     for pred in forecasts[3*i:3*i+3]]
            print(f"| {h} | {h-1:02}:00 | " + " | ".join(cells) + " |")
    (args.output / "summary.json").write_text(json.dumps([
        {"eve": DONORS[i//3], "hour1": case["known"][0]["viewers"], "forecast": direct[i]}
        for i, case in enumerate(cases)], indent=2) + "\n")
    print("Comparable shapes: hour-1 viewers and ratios at UTC 03, 05, 06, 10, 18, 23.")
    for launch in practices[0]["comparables"]:
        if launch["name"] not in ("ELDEN RING", "Cyberpunk 2077", "The Last of Us Part II",
                                   "The Legend of Zelda: Tears of the Kingdom"):
            continue
        case, truth = backtest.synthesize(launch, practices[0]["comparables"], "compare")
        p = case["known"][0]["viewers"]
        print(launch["name"], p, [round(truth[h] / p, 2) for h in (4, 6, 7, 11, 19, 24)])


if __name__ == "__main__":
    main()
