"""Exercise final minute-read glitches, DST boundaries and capacity ceilings.

Optionally compare healthy outputs byte for byte with an entry saved before
adding the robustness guard: --reference ../evidence/release-before-robustness.py
All persistent and temporary outputs are restricted to ../evidence/.
"""

import argparse
import copy
import json
import math
import statistics
import tempfile
from datetime import timedelta
from pathlib import Path

import backtest
import h00_scenarios
from scale_stress import EVIDENCE, ROOT, evidence_path, load_model


def reprint(case):
    """Rebuild the recorder's published hour-1 mean after a minute mutation."""
    rows = [r for r in case["history"] if case["day_start"] <= r["at"] < case["checkpoint"]]
    p = math.floor(statistics.mean(r["viewers"] for r in rows) + .5)
    channels = statistics.mean(r["channels"] for r in rows)
    case["known"][0].update(viewers=p, channels=channels, reads=len(rows))
    case["history_hourly"][-1].update(viewers=p, channels=channels, reads=len(rows))


def glitches(case):
    for label in ("zeros", "ten-minute-double", "partial-hour"):
        changed = copy.deepcopy(case)
        hour1 = [r for r in changed["history"] if case["day_start"] <= r["at"] < case["checkpoint"]]
        if label == "zeros":
            for r in hour1[::5]:
                r["viewers"] = r["channels"] = 0
        elif label == "ten-minute-double":
            for r in hour1[20:30]:
                r["viewers"] *= 2
                r["channels"] *= 2
        else:
            changed["history"] = [r for r in changed["history"] if r not in hour1[40:]]
        reprint(changed)
        yield label, changed


def check_dst(model):
    for date, expected in (("2026-11-19", 0), ("2023-10-20", 1), ("2022-11-09", 0),
                           ("2026-03-08T06:59:59", 0), ("2026-03-08T07:00:00", 1),
                           ("2026-11-01T05:59:59", 1), ("2026-11-01T06:00:00", 0)):
        stamp = date + ("T00:00:00" if "T" not in date else "") + "Z"
        assert model.us_dst(stamp) == expected, (stamp, expected)
    # A DST donor has its US spike at 04Z. A standard-time target needs 05Z.
    day = "2023-10-20T00:00:00Z"
    start = backtest.instant(day)
    launch = {"name": "DST clock probe", "day_start": day, "platforms": "PS5",
              "unlock": "rolling midnight local", "hourly": [
                  {"from": backtest.iso(start + timedelta(hours=h)), "viewers": 1000 if h == 4 else 100 + h}
                  for h in range(-24, 24)]}
    target = {"comparables": [launch], "day_start": "2026-11-19T00:00:00Z"}
    shifted = model.comparable_records(target)[0]
    target["day_start"] = day
    natural = model.comparable_records(target)[0]
    assert shifted["x"] == natural["x"], "DST changed hour-1/eve features"
    assert all(shifted["curve"][h] == natural["curve"][h] for h in (1, 2, 3))
    assert max(shifted["curve"], key=shifted["curve"].get) == 5
    assert max(natural["curve"], key=natural["curve"].get) == 4
    # Reverse the seasons: a standard donor's 05Z spike moves to 04Z.
    launch["day_start"] = "2022-11-09T00:00:00Z"
    launch["hourly"] = [{"from": backtest.iso(backtest.instant(launch["day_start"]) + timedelta(hours=h)),
                         "viewers": 1000 if h == 5 else 100 + h} for h in range(-24, 24)]
    reverse = model.comparable_records(target)[0]
    assert max(reverse["curve"], key=reverse["curve"].get) == 4


def check_sparse_history(model, case):
    removed = copy.deepcopy(case)
    start = backtest.instant(case["day_start"]) - timedelta(hours=5)
    end = start + timedelta(hours=1)
    removed["history_hourly"] = [r for r in removed["history_hourly"] if r["from"] != backtest.iso(start)]
    removed["history"] = [r for r in removed["history"] if not start <= backtest.instant(r["at"]) < end]
    sparse = copy.deepcopy(removed)
    sparse["history_hourly"].append({"from": backtest.iso(start), "viewers": 1e8, "channels": 10000, "reads": 29})
    sparse["history"].extend({"at": backtest.iso(start + timedelta(minutes=m)), "viewers": 1e8, "channels": 10000}
                             for m in range(29))
    for minute_only in (False, True):
        a, b = copy.deepcopy(removed), copy.deepcopy(sparse)
        if minute_only:
            a["history_hourly"] = b["history_hourly"] = None
        expected = model.forecast_case(a)
        assert model.forecast_case(b) == expected, "<30-read history hour affected the forecast"
        if minute_only:
            b["history"].append({"at": backtest.iso(start + timedelta(minutes=29)), "viewers": 1e8, "channels": 10000})
        else:
            b["history_hourly"][-1]["reads"] = 30
        assert model.forecast_case(b) != expected, "complete history hours were also discarded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=ROOT.parent / "gta6-twitch-ccu-forecast-01-kit")
    parser.add_argument("--answers", type=Path, default=EVIDENCE / "gta-practice-answers.json")
    parser.add_argument("--entry", type=Path, default=ROOT / "main.py")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--report", type=Path, default=EVIDENCE / "release-checks.json")
    args = parser.parse_args()
    report_path = evidence_path(args.report)
    model = load_model(args.entry)
    reference = load_model(args.reference) if args.reference else None
    check_dst(model)
    practices = [json.loads(p.read_text()) for p in sorted((args.kit / "input").glob("p*.in"))]
    template = backtest.final_shape(practices, practices[0]["comparables"], backtest.load_answers(args.answers))
    cases = [h00_scenarios.scenario(template, p, total, donor) for donor in h00_scenarios.DONORS
             for p, total in ((300000, 3600000), (1000000, 4000000), (2000000, 5000000))]
    h00_scenarios.check_scenarios(cases, args.entry)
    check_sparse_history(model, cases[4])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"dst": True, "capacity": True, "sparse_history": True, "glitches": [],
              "unprotected_control_movement": {},
              "healthy_byte_identical": None if args.reference is None else True}
    with tempfile.TemporaryDirectory(prefix="release-checks-", dir=EVIDENCE) as directory:
        temp = Path(directory)
        for i, case in enumerate(cases):
            healthy, elapsed, raw = backtest.run_entry(case, temp, args.entry)
            expected = model.forecast_case(case)
            assert json.loads(raw) == expected, "entry fell back"
            if args.reference:
                _, _, before = backtest.run_entry(case, temp, args.reference)
                assert before == raw, "healthy forecast changed after robustness guard"
            for label, changed in glitches(case):
                prediction, seconds, data = backtest.run_entry(changed, temp, args.entry)
                assert json.loads(data) == model.forecast_case(changed), "glitch triggered fallback"
                movement = max(abs(a / b - 1) for h, row in prediction.items() for a, b in zip(row, healthy[h]))
                assert movement < .03, (i, label, movement)
                if reference and i == 1:
                    unprotected = backtest.validate(json.dumps(reference.forecast_case(changed)).encode())
                    control = max(abs(a / b - 1) for h, row in unprotected.items() for a, b in zip(row, healthy[h]))
                    report["unprotected_control_movement"][label] = control
                    if label != "partial-hour":
                        assert control > .03, "glitch did not expose the old anchor's sensitivity"
                minute = copy.deepcopy(changed)
                minute["history_hourly"] = None
                assert model.forecast_case(minute) == model.forecast_case(changed), "glitch minute/hourly mismatch"
                report["glitches"].append({"eve": h00_scenarios.DONORS[i//3],
                    "hour1": case["known"][0]["viewers"], "glitch": label,
                    "max_quantile_movement": movement, "seconds": max(elapsed, seconds)})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("DST clock/boundary, capacity, missing-category, sparse-history and checkpoint checks passed.")
    print(f"Healthy byte comparison: {report['healthy_byte_identical']}; 12 scenarios, 36 injected glitches.")
    for label in ("zeros", "ten-minute-double", "partial-hour"):
        maximum = max(r["max_quantile_movement"] for r in report["glitches"] if r["glitch"] == label)
        print(f"{label}: maximum movement across every hour/quantile {maximum:.3%}")


if __name__ == "__main__":
    main()
