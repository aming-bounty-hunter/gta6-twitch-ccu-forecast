"""Run the real entry per case, validate it, and score held-out launch days.

Usage: python3 tools/backtest.py [--answers /path/to/answer.json] [--details]
Practice answers are not shipped in the kit. Missing labels are reported, never
replaced with a guess. All synthetic files live in a temporary directory.
"""

import argparse
import copy
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)


def instant(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def validate(data):
    if len(data) > 65536:
        raise ValueError("output exceeds 65536 bytes")
    value = json.loads(data)
    if not isinstance(value, dict) or set(value) != {"hours"}:
        raise ValueError("wrong root keys")
    rows = value["hours"]
    if not isinstance(rows, list) or len(rows) != 23:
        raise ValueError("need 23 hours")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"hour", "q"}:
            raise ValueError("wrong row keys")
        h, q = row["hour"], row["q"]
        if type(h) is not int or h not in range(2, 25) or h in seen:
            raise ValueError("invalid/duplicate hour")
        seen.add(h)
        if not isinstance(q, list) or len(q) != 5:
            raise ValueError("need five quantiles")
        if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 < v <= 1e9 for v in q):
            raise ValueError("invalid quantile")
        if q != sorted(q):
            raise ValueError("crossed quantiles")
    return {row["hour"]: row["q"] for row in rows}


def persistence(p):
    p = max(1, p)
    return {h: [math.floor(p * math.exp(z * .15 * math.sqrt(h - 1)) * 100 + .5) / 100
                for z in Z] for h in range(2, 25)}


def score(forecast, truth):
    losses, errors, covers = [], [], []
    for h, y in sorted(truth.items()):
        y = max(1, y)
        q = forecast[h]
        for tau, v in zip(LEVELS, q):
            d = math.log(y) - math.log(v)
            losses.append(2 * max(tau * d, (tau - 1) * d))
        errors.append(abs(math.log(y) - math.log(q[2])))
        covers.append(q[0] <= y <= q[4])
    if not losses:
        raise ValueError("no scored hours")
    return {"loss": statistics.mean(losses), "abs_log": statistics.mean(errors),
            "coverage": statistics.mean(covers), "n": len(truth)}


def run_entry(case, temp, entry, raw=None):
    input_dir, output_dir = temp / "input", temp / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    for path in input_dir.glob("*.in"):
        path.unlink()
    for path in output_dir.glob("*.out"):
        path.unlink()
    case_id = case["id"]
    (input_dir / (case_id + ".in")).write_text(json.dumps(case) if raw is None else raw, encoding="utf-8")
    elapsed = invoke_entry(input_dir, output_dir, entry)
    output = output_dir / (case_id + ".out")
    if output.is_symlink() or not output.is_file():
        raise ValueError("missing regular output file")
    data = output.read_bytes()
    return validate(data), elapsed, data


def invoke_entry(input_dir, output_dir, entry):
    env = dict(os.environ, MAKEFUN_INPUT_DIR=str(input_dir), MAKEFUN_OUTPUT_DIR=str(output_dir), PYTHONHASHSEED="0")
    started = time.perf_counter()
    process = subprocess.run([sys.executable, "-P", "-s", str(entry)], env=env,
                             cwd=ROOT, capture_output=True, timeout=8)
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise ValueError(f"entry exit {process.returncode}: {process.stderr.decode()[:500]}")
    return elapsed


def grade_entry(case, temp, entry):
    """An invalid execution gets flat hour-1 quantiles, as the contest specifies."""
    started = time.perf_counter()
    try:
        pred, elapsed, data = run_entry(case, temp, entry)
        return pred, elapsed, data, True, None
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        p = max(1, case["known"][0]["viewers"])
        pred = {h: [p] * 5 for h in range(2, 25)}
        return pred, time.perf_counter() - started, None, False, str(error)


def is_target_history(launch):
    text = launch.get("unlock", "").lower()
    return "target-history" in text or "not a launch" in text


def gta_like(launch):
    text = launch.get("unlock", "").lower()
    platforms = launch.get("platforms", "").lower()
    # Include explicitly mixed console/PC rolling launches, but exclude early access.
    return ("midnight local" in text or "12 am local" in text or "rolling" in text) and (
        any(k in platforms for k in ("ps", "xbox", "switch"))) and (
        "early-access-contaminated" not in text)


def console_only(launch):
    platforms = launch.get("platforms", "").lower().split("(")[0]
    return "pc" not in platforms and "stadia" not in platforms


def synthesize(launch, comparables, case_id):
    start = instant(launch["day_start"])
    points = {}
    for point in launch["hourly"]:
        h = int((instant(point["from"]) - start).total_seconds() // 3600)
        points[h] = point
    if 0 not in points:
        return None
    truth = {h + 1: point["viewers"] for h, point in points.items() if 1 <= h <= 23}
    if len(truth) < 12:
        return None
    history = [{"at": p["from"], "viewers": p["viewers"], "channels": p.get("channels")}
               for h, p in sorted(points.items()) if h <= 0]
    hourly = [{"from": p["at"], "viewers": p["viewers"], "channels": p["channels"], "reads": 1}
              for p in history]
    known = dict(points[0], hour=1, until=iso(start + timedelta(hours=1)), reads=1)
    case = {"schema": "makefun/twitch-ccu-case/v1", "id": case_id,
            "game": {"name": launch["name"], "twitch_id": "synthetic"},
            "day_start": iso(start), "checkpoint": iso(start + timedelta(hours=1)),
            "levels": list(LEVELS), "known": [known],
            "forecast": [{"hour": h, "from": iso(start + timedelta(hours=h - 1)),
                          "until": iso(start + timedelta(hours=h))} for h in range(2, 25)],
            "history": history, "history_hourly": hourly, "streams": None, "categories_hourly": None,
            "comparables": [a for a in comparables if a["name"] != launch["name"]],
            "notes": f"{launch['name']} on {launch.get('platforms', '')}; real unlock: {launch.get('unlock', '')}"}
    return case, truth


def final_shape(practices, launches, answers):
    case = copy.deepcopy(practices[0])
    case["id"] = "final_shape"
    case["game"] = {"name": "Grand Theft Auto VI", "twitch_id": "1867279146"}
    case["notes"] = "PS5 and Xbox Series; rolling midnight local. US east 05:00Z, west 08:00Z."
    case["comparables"] = list(launches)
    # The real final adds these four launches to its training pool. Use the
    # independently fetched labels, when available, to exercise all 41 donors.
    for practice in practices:
        hourly = [{"from": p["at"], "viewers": p["viewers"], "channels": p.get("channels")}
                  for p in practice["history"]]
        start = instant(practice["day_start"])
        hourly.extend({"from": iso(start + timedelta(hours=h - 1)), "viewers": y, "channels": None}
                      for h, y in answers.get(practice["id"], {}).items())
        case["comparables"].append({"name": practice["game"]["name"], "day_start": practice["day_start"],
            "platforms": "PS5, Xbox", "unlock": practice["notes"],
            "hourly": hourly})
    shift = instant("2026-11-19T00:00:00Z") - instant(case["day_start"])
    case["day_start"] = iso(instant(case["day_start"]) + shift)
    case["checkpoint"] = iso(instant(case["checkpoint"]) + shift)
    for row in case["known"] + case["forecast"]:
        row["from"] = iso(instant(row["from"]) + shift)
        row["until"] = iso(instant(row["until"]) + shift)
        if "viewers" in row:
            row["viewers"] *= 80
        if row.get("channels") is not None:
            row["channels"] *= 20
    for row in case["history_hourly"]:
        row["from"] = iso(instant(row["from"]) + shift)
        row["viewers"] *= 80
        if row.get("channels") is not None:
            row["channels"] *= 20
    history = []
    cutoff = instant(case["checkpoint"]) - timedelta(hours=48)
    for row in case["history_hourly"]:
        t = instant(row["from"])
        if t < cutoff:
            continue
        for minute in range(60):
            # Symmetric integer changes keep the exact hourly mean unchanged.
            step = int(row["viewers"] // 1000)
            history.append({"at": iso(t + timedelta(minutes=minute)),
                            "viewers": row["viewers"] + step * (2 * minute - 59),
                            "channels": row.get("channels")})
        row["reads"] = 60
    case["history"] = history
    case["history_hourly"] = [r for r in case["history_hourly"] if instant(r["from"]) >= cutoff]
    case["known"][0]["reads"] = 60
    p1 = case["known"][0]["viewers"]
    case["streams"] = [{"login": f"stream{i:02}", "viewers": max(1, int(.18 * p1 / (i + 1))),
                        "language": ("en", "de", "es", "ja")[i % 4],
                        "started_at": iso(instant(case["day_start"]) - timedelta(hours=4))} for i in range(20)]
    case["categories_hourly"] = [{"from": r["from"], "top": [
        {"name": f"Category {i}", "viewers": 1000000 // (i + 1), "channels": 10000 // (i + 1)}
        for i in range(20)]} for r in case["history_hourly"]]
    return case


def load_answers(path):
    if not path:
        return {}
    value = json.loads(path.read_text())
    cases = value.get("cases", value)
    if isinstance(cases, list):
        cases = {c["id"]: c for c in cases}
    result = {}
    for name, case in cases.items():
        rows = case.get("hours", case) if isinstance(case, dict) else case
        if isinstance(rows, dict):
            result[name] = {int(h): y for h, y in rows.items() if 2 <= int(h) <= 24}
        else:
            result[name] = {r["hour"]: r["viewers"] for r in rows if 2 <= r["hour"] <= 24}
    return result


def aggregate(rows):
    if not rows:
        return None
    n = sum(r["n"] for r in rows)
    return {k: sum(r[k] * r["n"] for r in rows) / n for k in ("loss", "abs_log", "coverage")}


def print_row(label, model, house, elapsed):
    if model is None:
        print(f"{label:30} {'LABELS MISSING':>12} {'--':>10} {'--':>9} {'--':>10} {elapsed * 1000:8.1f}")
    else:
        miss = 100 * math.expm1(model["abs_log"])
        print(f"{label:30} {model['loss']:12.6f} {house['loss']:10.6f} {miss:8.2f}% {model['coverage']:9.3f} {elapsed * 1000:8.1f}")


def self_check():
    sample = {"hours": [{"hour": h, "q": [600000, 800000, 900000, 1100000, 1500000]} for h in range(2, 25)]}
    result = score(validate(json.dumps(sample).encode()), {2: 1000000, 3: 700000})
    assert abs(result["loss"] - 0.112545) < .000001, result
    for bad in (
        {"hours": sample["hours"][:-1]},
        {"hours": [sample["hours"][0]] * 23},
        {"hours": [{"hour": h, "q": [1, 2, 1, 4, 5]} for h in range(2, 25)]},
        {"hours": [{"hour": h, "q": [0, 1, 2, 3, 4]} for h in range(2, 25)]},
    ):
        try:
            validate(json.dumps(bad).encode())
        except ValueError:
            pass
        else:
            raise AssertionError("validator accepted a corrupted output")
    altered = copy.deepcopy(sample)
    altered["hours"][0]["q"] = [x * 2 for x in altered["hours"][0]["q"]]
    assert score(validate(json.dumps(altered).encode()), {2: 1000000, 3: 700000})["loss"] > result["loss"]
    # Swap hour assignments with the same aggregate values: scoring must still change.
    swapped = copy.deepcopy(altered)
    swapped["hours"][0]["q"], swapped["hours"][1]["q"] = swapped["hours"][1]["q"], swapped["hours"][0]["q"]
    assert score(validate(json.dumps(swapped).encode()), {2: 1000000, 3: 700000}) != score(
        validate(json.dumps(altered).encode()), {2: 1000000, 3: 700000})


def check_press(entry, launches):
    """Compare analytic calibration residuals with literal held-out refitting."""
    spec = importlib.util.spec_from_file_location("checked_entry", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = module.comparable_records({"comparables": launches})[:9]
    _, errors = module.fit_regression(records)
    index = 2
    coefficients, _ = module.fit_regression(records[:index] + records[index + 1:])
    x = module.regression_features(records[index]["x"])
    for h, y in records[index]["curve"].items():
        literal = y - sum(a * b for a, b in zip(x, coefficients[h - 1]))
        if abs(literal - errors[index][h]) > 1e-8:
            raise AssertionError((h, literal, errors[index][h]))


def robustness_checks(final, temp, entry):
    original, _, _ = run_entry(final, temp, entry)
    minute_only = copy.deepcopy(final)
    minute_only["history_hourly"] = None
    minute_prediction, _, _ = run_entry(minute_only, temp, entry)
    if minute_prediction != original:
        raise AssertionError("minute aggregation disagrees with exact hourly means")
    # Odd inputs must produce the same valid persistence fallback shape.
    probes = ["{broken JSON", "null", json.dumps({"known": [{"hour": 1, "viewers": 1e300}]}),
              json.dumps({"day_start": "bad timestamp", "known": [{"hour": 1, "viewers": 12345}]}),
              json.dumps({"day_start": final["day_start"], "known": [{"hour": 1, "viewers": 0}],
                          "history": [{"at": None, "viewers": None}], "comparables": [None]})]
    for raw in probes:
        run_entry({"id": "odd_input"}, temp, entry, raw=raw)
    pred, _, _ = run_entry({"id": "preserve_print"}, temp, entry, raw=probes[3])
    if pred != persistence(12345):
        raise AssertionError("fallback lost the available hour-1 print")
    input_dir, output_dir = temp / "input", temp / "output"
    for path in input_dir.glob("*.in"):
        path.unlink()
    for path in output_dir.glob("*.out"):
        path.unlink()
    # A broken first file must not prevent later files from being processed.
    payloads = {"z_valid": json.dumps({"known": [{"hour": 1, "viewers": 23456}]}),
                "a_invalid": "invalid JSON", "m_valid": json.dumps({"known": [{"hour": 1, "viewers": 34567}]})}
    for name, raw in payloads.items():
        (input_dir / (name + ".in")).write_text(raw)
    invoke_entry(input_dir, output_dir, entry)
    if sorted(p.stem for p in output_dir.glob("*.out")) != sorted(payloads):
        raise AssertionError("batch did not process every input")
    for name, p in (("a_invalid", 1), ("m_valid", 34567), ("z_valid", 23456)):
        if validate((output_dir / (name + ".out")).read_bytes()) != persistence(p):
            raise AssertionError(f"batch fallback mismatch: {name}")
    invalid, _, _, valid, _ = grade_entry(final, temp, Path(os.devnull))
    p = final["known"][0]["viewers"]
    if valid or invalid != {h: [p] * 5 for h in range(2, 25)}:
        raise AssertionError("grader did not assign the specified invalid-output penalty")
    return len(probes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=ROOT.parent / "gta6-twitch-ccu-forecast-01-kit")
    parser.add_argument("--entry", type=Path, default=ROOT / "main.py")
    parser.add_argument("--answers", type=Path, default=ROOT.parent / "evidence/gta-practice-answers.json",
                        help="cached practice labels (default: ../evidence/gta-practice-answers.json)")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--report", type=Path, help="optional JSON results; keep outside the repo")
    args = parser.parse_args()
    self_check()
    practices = [json.loads(p.read_text()) for p in sorted((args.kit / "input").glob("p*.in"))]
    if len(practices) != 4:
        raise ValueError("expected four practice cases")
    answer_path = args.answers
    answers = load_answers(answer_path)
    launches = practices[0]["comparables"]
    press_checked = args.entry.resolve() == (ROOT / "main.py").resolve()
    if press_checked:
        check_press(args.entry, launches)
    report = {"practice": [], "loo": [], "skipped": []}
    print(f"{'case / set':30} {'loss':>12} {'baseline':>10} {'miss':>9} {'90% cover':>10} {'run ms':>8}")
    with tempfile.TemporaryDirectory(prefix="gta-backtest-") as directory:
        temp = Path(directory)
        for case in practices:
            pred, elapsed, data, valid, error = grade_entry(case, temp, args.entry)
            _, elapsed2, data2, valid2, error2 = grade_entry(case, temp, args.entry)
            deterministic = valid and valid2 and data == data2
            truth = answers.get(case["id"])
            model = score(pred, truth) if truth else None
            house = score(persistence(case["known"][0]["viewers"]), truth) if truth else None
            label = case["id"] + " " + case["game"]["name"][:23]
            print_row(label, model, house, elapsed)
            if not valid or not valid2 or not deterministic:
                print(f"  INVALID / UNRANKED: {error or error2 or 'nondeterministic output'}")
            report["practice"].append({"id": case["id"], "model": model, "baseline": house,
                "seconds": max(elapsed, elapsed2), "valid": valid and valid2, "deterministic": deterministic})
        pmodel = aggregate([r["model"] for r in report["practice"] if r["model"]])
        phouse = aggregate([r["baseline"] for r in report["practice"] if r["baseline"]])
        print_row("PRACTICE mean", pmodel, phouse, sum(r["seconds"] for r in report["practice"]) / 4)
        for i, launch in enumerate(launches):
            if is_target_history(launch):
                continue
            result = synthesize(launch, launches, f"loo{i:02}")
            if result is None:
                report["skipped"].append(launch["name"])
                continue
            case, truth = result
            assert all(a["name"] != launch["name"] for a in case["comparables"])
            pred, elapsed, _, valid, error = grade_entry(case, temp, args.entry)
            model = score(pred, truth)
            house = score(persistence(case["known"][0]["viewers"]), truth)
            row = {"name": launch["name"], "gta_like": gta_like(launch), "model": model,
                   "console_only": console_only(launch), "baseline": house, "seconds": elapsed,
                   "valid": valid, "error": error}
            report["loo"].append(row)
            if args.details:
                print_row(("* " if row["gta_like"] else "  ") + launch["name"][:28], model, house, elapsed)
        for label, rows in (("LOO all", report["loo"]),
                            ("LOO console / rolling", [r for r in report["loo"] if r["gta_like"]]),
                            ("LOO console-only / rolling", [r for r in report["loo"] if r["gta_like"] and r["console_only"]])):
            print_row(f"{label} (n={len(rows)})", aggregate([r["model"] for r in rows]),
                      aggregate([r["baseline"] for r in rows]), statistics.mean(r["seconds"] for r in rows))
        final = final_shape(practices, launches, answers)
        _, elapsed, data = run_entry(final, temp, args.entry)
        _, elapsed2, data2 = run_entry(final, temp, args.entry)
        if data != data2:
            raise ValueError("final-shaped case is nondeterministic")
        report["final_shape"] = {"deterministic": True, "seconds": max(elapsed, elapsed2),
                                 "bytes": len(data), "history_points": len(final["history"]),
                                 "comparables": len(final["comparables"])}
        print(f"Final shape: valid, deterministic, {max(elapsed, elapsed2):.3f}s, {len(data)} bytes; "
              f"{len(final['history'])} minute reads, {len(final['comparables'])} comparables")
        if press_checked:
            probes = robustness_checks(final, temp, args.entry)
            report["robustness"] = {"odd_inputs_valid": probes, "minute_hourly_equivalent": True,
                                     "press_verified": True, "batch_valid": True}
            print(f"Robustness: {probes} odd inputs valid; batch valid; minute/hourly aggregation agrees; "
                  "PRESS matches literal LOO")
        report["valid"] = all(r["valid"] for r in report["practice"] + report["loo"])
        report["deterministic"] = all(r["deterministic"] for r in report["practice"])
        print(f"Valid: {int(report['valid'])}; deterministic practice/final: {int(report['deterministic'])}")
    if report["skipped"]:
        print("Skipped (missing hour 1 or <12 answer hours):", ", ".join(report["skipped"]))
    if not all(r["model"] for r in report["practice"]):
        print("Practice scores INCOMPLETE: supply --answers; no labels were inferred.")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not all(r["model"] for r in report["practice"]):
        raise SystemExit(2)
    if not report["valid"] or not report["deterministic"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
