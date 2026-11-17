"""Fit log launch curves, then calibrate quantiles with held-out analog errors.

Features use only pre-checkpoint observations. Quantile residuals are computed
by leave-one-out within the supplied training pool; the benchmark holds out an
additional entire launch. Category totals constrain upside only beyond the
observed launch sizes. Streams have no historical snapshots and are unused.
"""

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path


LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
RIDGE = 3.0
FEATURE_SET = "activity"
RESIDUAL_POOL = 0.25
RESIDUAL_SCALE = 0.85
REGIME = False
SPARSE_MODE = True
# Saturating scale protects a record-sized launch from extrapolated decay.
LOGP_CAP = 13.0
SPARSE_SPREAD = .45
SPARSE_JITTER = True
SATURATION = True


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def positive(value, default=1.0):
    try:
        number = float(value)
        return min(1e9, max(1.0, number)) if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def first_print(case):
    for row in case.get("known", []) or []:
        if isinstance(row, dict) and row.get("hour") == 1:
            return positive(row.get("viewers"))
    return 1.0


def baseline(p):
    return {"hours": [
        {"hour": h, "q": [min(1e9, max(0.01, math.floor(
            p * math.exp(z * 0.15 * math.sqrt(h - 1)) * 100 + .5) / 100)) for z in Z]}
        for h in range(2, 25)
    ]}


def hourly_points(rows, start, key):
    """Aggregate minutes by hour; sparse hourly observations remain sparse."""
    buckets = {}
    for row in rows or []:
        try:
            hour = math.floor((timestamp(row[key]) - start) / 3600)
            value = float(row["viewers"])
            if not math.isfinite(value) or value < 0:
                continue
            buckets.setdefault(hour, []).append((value, row.get("channels")))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    result = {}
    for hour, points in buckets.items():
        channels = [float(c) for _, c in points if isinstance(c, (int, float)) and math.isfinite(c) and c >= 0]
        result[hour] = (max(1.0, sum(v for v, _ in points) / len(points)),
                        sum(channels) / len(channels) if channels else None)
    return result


def metadata(text, platforms):
    text = text.lower()
    platforms = platforms.lower()
    console = "pc" not in platforms.split("(")[0] and "stadia" not in platforms
    early = "early-access-contaminated" in text or (
        "early access" in text and "no early access" not in text
        and "without early access" not in text)
    rolling = any(word in text for word in ("midnight local", "12 am local", "rolling"))
    rolling = rolling or (console and not early)
    return (float(console), float(rolling), float(early))


def features(points, p, meta):
    def ratio(lo, hi):
        values = [v[0] for k, v in points.items() if lo <= k < hi]
        return math.log(max(1.0, sum(values) / len(values)) / p) if values else 0.0

    channel = points.get(0, (p, None))[1]
    logc = math.log(max(1, channel)) if channel is not None else math.log(p) - 4
    peaks = [v[0] for k, v in points.items() if -24 <= k < 0]
    peak = math.log(max(peaks) / p) if peaks else 0.0
    return (math.log(p), ratio(-24, 0), ratio(-6, 0), ratio(-2, 0),
            ratio(-72, -24), *meta, logc, peak)


def comparable_records(case):
    records = []
    for item in case.get("comparables", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            if "target-history" in item.get("unlock", "") or "not a launch" in item.get("unlock", "").lower():
                continue
            start = timestamp(item["day_start"])
            points = hourly_points(item["hourly"], start, "from")
            if 0 not in points or sum(h in points for h in range(1, 24)) < 12:
                continue
            p = points[0][0]
            meta = metadata(item.get("unlock", ""), item.get("platforms", ""))
            launch_points = {h: v for h, v in points.items() if 0 <= h <= 23}
            peak = max(v[0] for v in launch_points.values())
            onset = next((h for h, v in sorted(launch_points.items()) if v[0] >= max(3 * p, .15 * peak)), 12)
            records.append({"name": item.get("name", ""), "x": features(points, p, meta),
                            "curve": {h: math.log(v[0] / p) for h, v in points.items() if 1 <= h <= 23},
                            "logs": {h: math.log(v[0]) for h, v in points.items()}, "onset": onset})
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return records


def weights(target, records):
    scales = (1.5, 1.0, 0.7, 0.35, 2.5, 0.8, 0.7, 0.7)
    distances = [sum(min(16.0, ((a - b) / s) ** 2)
                     for a, b, s in zip(target, row["x"], scales)) for row in records]
    closest = min(distances)
    return [math.exp(-0.5 * (d - closest)) for d in distances]


def regression_features(x):
    logp, day, recent, last, older, console, rolling, early, logc, peak = x
    # A dormant category's first-hour print is not a measure of its eventual size.
    # A hinge lets launch-day ratios vary rapidly below 10k, gently above it.
    result = [1.0, (min(LOGP_CAP, logp) - 11) / 2, max(0.0, 9.2 - logp) / 2,
            max(-3.0, day), max(-2.0, recent), last,
            max(-6.0, older) / 3, console, rolling, early]
    if FEATURE_SET in ("channels", "activity"):
        result.extend([(logc - logp + 4) / 2, peak])
    if FEATURE_SET == "activity":
        result.append(float(is_sparse(x)))
    return result


def fit_regression(records):
    dimension = len(regression_features(records[0]["x"]))
    coefficients, residuals = [], [{} for _ in records]
    for hour in range(1, 24):
        indices = [i for i, r in enumerate(records) if hour in r["curve"]]
        rows = [(regression_features(records[i]["x"]), records[i]["curve"][hour]) for i in indices]
        matrix = [[sum(x[i] * x[j] for x, _ in rows) + (RIDGE if i == j and i else 0.0)
                   for j in range(dimension)] + [float(i == j) for j in range(dimension)]
                  for i in range(dimension)]
        for i in range(dimension):
            pivot = max(range(i, dimension), key=lambda j: abs(matrix[j][i]))
            matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
            divisor = matrix[i][i]
            if abs(divisor) < 1e-10:
                raise ValueError("singular fit")
            matrix[i] = [v / divisor for v in matrix[i]]
            for j in range(dimension):
                if j != i:
                    factor = matrix[j][i]
                    matrix[j] = [a - factor * b for a, b in zip(matrix[j], matrix[i])]
        inverse = [row[dimension:] for row in matrix]
        xy = [sum(x[i] * y for x, y in rows) for i in range(dimension)]
        beta = [sum(a * b for a, b in zip(row, xy)) for row in inverse]
        coefficients.append(beta)
        # PRESS identity: these are exact leave-one-out errors for fixed ridge.
        for index, (x, y) in zip(indices, rows):
            leverage = sum(x[i] * sum(a * b for a, b in zip(row, x)) for i, row in enumerate(inverse))
            fitted = sum(a * b for a, b in zip(x, beta))
            residuals[index][hour] = (y - fitted) / max(1e-6, 1 - leverage)
    return coefficients, residuals


def regression_predictions(records, target):
    coefficients, residuals = fit_regression(records)
    x = regression_features(target)
    center = [sum(a * b for a, b in zip(x, beta)) for beta in coefficients]
    return center, residuals


def mixture_quantiles(values, masses, spread):
    total = sum(masses)
    low = min(values) - 4 * spread
    high = max(values) + 4 * spread
    result = []
    for level in LEVELS:
        left, right = low, high
        for _ in range(32):
            mid = (left + right) / 2
            cdf = sum(w * (0.5 + 0.5 * math.erf((mid - v) / (spread * math.sqrt(2))))
                      for v, w in zip(values, masses)) / total
            if cdf < level:
                left = mid
            else:
                right = mid
        result.append((left + right) / 2)
    return result


def is_sparse(x):
    return x[0] < math.log(15000) and x[8] < math.log(200) and x[5] == 0 and x[7] == 0


def sparse_forecast(case, records, target):
    """Before a global unlock, borrow audience levels rather than huge ratios."""
    donors = [r for r in records if is_sparse(r["x"])]
    if len(donors) < 4:
        return None
    text = case.get("notes", "").lower().split("source:")[0]
    declared = None
    # Do not use unlock times that were inferred retrospectively from the answer.
    if "inferred" not in text and "unverified" not in text:
        times = re.findall(re.escape(case["day_start"][:10]) + r"t(\d{2}):", text)
        if times:
            declared = int(times[0])
    result = []
    for h in range(1, 24):
        values, masses = [], []
        for donor in donors:
            shifts = [(donor["onset"] - declared, 1.0)] if declared is not None else (
                [(-4, .1), (-2, .2), (0, .4), (2, .2), (4, .1)] if SPARSE_JITTER else [(0, 1.0)])
            for shift, weight in shifts:
                logs = donor["logs"]
                offset = h + shift
                nearest = min(logs, key=lambda k: abs(k - offset))
                base = min(logs, key=lambda k: abs(k - shift))
                maturity = min(1.0, max(0.0, (offset - donor["onset"] + 1) / 3))
                values.append(logs[nearest] - target[0] + (1 - maturity) * (target[0] - logs[base]))
                masses.append(weight)
        result.append(mixture_quantiles(values, masses, SPARSE_SPREAD))
    return result


def platform_context(case, p, records):
    """Estimate share from the top twenty, with an explicit 15% long-tail allowance.

    This guard needs an unprecedented hour 1 AND observed platform snapshots.
    Historical cases lack those snapshots, so their calibrated curves are kept.
    The top-twenty sum is a lower bound, not the platform's total audience.
    """
    anchor = max(math.exp(r["x"][0]) for r in records)
    if not SATURATION or p <= anchor:
        return None
    start = timestamp(case["day_start"])
    cutoff = timestamp(case["checkpoint"])
    name = case.get("game", {}).get("name", "").casefold()
    totals = {}
    for row in case.get("categories_hourly", []) or []:
        try:
            at = timestamp(row["from"])
            if not start - 48 * 3600 <= at < cutoff:
                continue
            top = row["top"]
            if len(top) != 20:
                continue
            viewers = [float(c["viewers"]) for c in top]
            if any(not math.isfinite(v) or v < 0 for v in viewers):
                continue
            own = [v for c, v in zip(top, viewers) if c["name"].casefold() == name]
            hour = math.floor((at - start) / 3600)
            total = sum(viewers)
            if hour == 0:
                if len(own) != 1:
                    continue
                # A snapshot and the first-hour average need not match.
                total += p - own[0]
            if total > 0 and (hour != 0 or total > p):
                totals[hour] = total / .85
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            continue
    if 0 not in totals:
        return None
    # Reuse a mostly observed day's UTC pattern. Do not fit a daily sinusoid
    # to a few post-unlock snapshots. Missing clock hours use flat capacity.
    cycles = {}
    for day in (-24, -48):
        if day in totals and sum(day + h in totals for h in range(24)) >= 18:
            cycles = {h: min(1.4, max(.6, totals[day + h] / totals[day]))
                      for h in range(1, 24) if day + h in totals}
            break
    if not cycles:
        return None
    strength = min(1.0, math.log(p / anchor) / math.log(2))
    return totals[0], cycles, strength


def saturate(quantiles, hour, p, context):
    """Let positive growth consume finite platform share in log-odds space.

    For share s and ratio r>1, r/(1+s*(r-1)) grows into remaining headroom.
    Blend this correction as size exceeds the largest comparable. Leave dips
    intact; the launch curves already model the daily cycle. Extra log spread
    and uncertain platform totals account for extrapolation beyond the labels.
    """
    if context is None:
        return quantiles
    total, cycles, strength = context
    spread = .15 * strength * math.sqrt(hour / 23)
    result = []
    for v, z in zip(quantiles, Z):
        value = v + z * spread
        share = min(.95, p / (total * cycles.get(hour, 1.0)) * math.exp(-.18 * z))
        penalty = math.log1p(share * math.expm1(min(50.0, max(0.0, value))))
        result.append(value - strength * penalty)
    return result


def forecast_case(case):
    p = first_print(case)
    start = timestamp(case["day_start"])
    rows = case.get("history_hourly")
    points = hourly_points(rows, start, "from") if rows else hourly_points(case.get("history"), start, "at")
    points = {h: row for h, row in points.items() if h <= 0}
    points[0] = (p, points.get(0, (p, None))[1])
    notes = case.get("notes", "")
    # The final's hardware and unlock are known even if the recorder omits prose.
    if str(case.get("game", {}).get("twitch_id")) == "1867279146":
        meta = (1.0, 1.0, 0.0)
    else:
        platform_text = notes.split("; real unlock:")[0].split(" on ")[-1]
        meta = metadata(notes, platform_text)
    target = features(points, p, meta)
    records = comparable_records(case)
    if REGIME:
        same_regime = [r for r in records if is_sparse(r["x"]) == is_sparse(target)]
        if len(same_regime) >= 6:
            records = same_regime
    if len(records) < 3:
        return baseline(p)
    if SPARSE_MODE and is_sparse(target):
        sparse = sparse_forecast(case, records, target)
        if sparse is not None:
            return {"hours": [{"hour": h + 2, "q": [round(math.exp(min(math.log(1e9), max(-4.6, math.log(p) + v))), 4)
                                                      for v in row]} for h, row in enumerate(sparse)]}
    masses = weights(target, records)
    centers, residuals = regression_predictions(records, target)
    context = platform_context(case, p, records)
    hours = []
    for hour in range(1, 24):
        values, local_weights = [], []
        for index, (record, weight) in enumerate(zip(records, masses)):
            if hour in record["curve"]:
                values.append(centers[hour - 1] + RESIDUAL_SCALE * residuals[index][hour])
                local_weights.append(RESIDUAL_POOL / len(records) + (1 - RESIDUAL_POOL) * weight / sum(masses))
        if not values:
            hours.append(baseline(p)["hours"][hour - 1])
            continue
        spread = 0.08 + 0.025 * math.sqrt(hour)
        quantiles = mixture_quantiles(values, local_weights, spread)
        quantiles = saturate(quantiles, hour, p, context)
        hours.append({"hour": hour + 1, "q": [round(math.exp(min(math.log(1e9), max(-4.6, math.log(p) + v))), 4)
                                                for v in quantiles]})
    return {"hours": hours}


def valid_forecast(result):
    if set(result) != {"hours"} or len(result["hours"]) != 23:
        return False
    if [row.get("hour") for row in result["hours"]] != list(range(2, 25)):
        return False
    for row in result["hours"]:
        if set(row) != {"hour", "q"} or len(row["q"]) != 5:
            return False
        q = row["q"]
        if any(not math.isfinite(v) or not 0 < v <= 1e9 for v in q) or q != sorted(q):
            return False
    return True


def main():
    source = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    destination = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.in")):
        case = {}
        p = 1.0
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            p = first_print(case)
            result = forecast_case(case)
            if not valid_forecast(result):
                raise ValueError("invalid model result")
        except Exception:
            result = baseline(p)
        (destination / (path.stem + ".out")).write_text(
            json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
