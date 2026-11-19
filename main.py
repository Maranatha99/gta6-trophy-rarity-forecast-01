"""Forecast GTA VI PS5 trophy earn rates on Exophase.

Entry for make.fun gta6-trophy-rarity-forecast-01. Reads one case file
(schema makefun/gta6-trophies-case/v1) per run from $MAKEFUN_INPUT_DIR
(or /input/) and writes one forecast file per case to $MAKEFUN_OUTPUT_DIR
(or /output/): five quantiles of each trophy's earn rate at the target.

Method, per trophy
------------------
Two independent median models are fitted on the case history and blended
in logit space:

1. Logit-trend model. A least-squares line of logit(rate) against
   ln(1 + hours since 2026-11-19T00:00Z), fitted over the last 48 hours
   of reads (all reads if fewer). The slope is shrunk toward the median
   slope of the trophy's tier (platinum / gold / silver / bronze), because
   trophies of a tier move together as the player base grows and plays.
   The line is read at the target instant.

2. Earner-count model. Earners E = rate * players_tracked / 100 and the
   player count P are each fitted as log-linear in the same time variable
   (slopes shrunk toward their tier medians for earners). Projecting both
   to the target and dividing separates the two forces that move a rate:
   new players dilute it, existing players earning trophies raise it.
   This matters because Exophase refreshes its counts in jumps: a refresh
   that multiplies the tracked players dilutes late-game trophies far
   more than the opening trophy everyone already holds.

Spread. The predictive spread in logit units combines a random-walk term
(0.5 * sqrt(horizon in days)), the residual scatter of the logit fit,
the scatter of logit jumps between distinct reads (refresh shocks), and
a floor that grows with the horizon so a flat launch-night history still
admits that a refresh can move every rate. Quantiles are the median
shifted by the standard-normal z for each level, mapped back to percent.

Edge cases: rates are clamped to [0.01, 99.99] before any logit; null
rates are skipped; with fewer than 3 usable reads a trophy falls back to
its last rate with the random-walk-plus-floor spread. No network, no
third-party packages, deterministic: the same case always yields the
same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.6448536269514722, -0.6734897501960817, 0.0,
            0.6734897501960817, 1.6448536269514722]

# Time origin for the trend variable: the release day, 2026-11-19 00:00 UTC.
ORIGIN = datetime(2026, 11, 19, tzinfo=timezone.utc)
# How far back the fit looks, in hours.
FIT_WINDOW_HOURS = 48.0
# How much of a trophy's own slope survives shrinkage to its tier median.
OWN_SLOPE_WEIGHT = 0.6
# Blend weight of the logit-trend median vs the earner-count median.
TREND_WEIGHT = 0.5
# Random-walk scale, logit units per sqrt(day).
WALK_SIGMA = 0.5
# Spread floor: a flat history must still admit a refresh shock.
FLOOR_BASE = 0.30
FLOOR_PER_SQRT_DAY = 0.10
# Hard bounds for the predictive spread, in logit units.
SPREAD_MIN = 0.35
SPREAD_MAX = 3.0
# Clamp for rates before the logit (the page prints 2 decimals).
RATE_LO = 0.01
RATE_HI = 99.99
# How much a never-refreshed page is assumed to fall at the next refresh,
# in logit units. Exophase refreshes its counts in jumps: a refresh only
# dilutes rates (the newly tracked players hold fewer trophies each), so
# when the whole observed history is one unchanged page, the target lies
# past a refresh the data has never shown. -0.30 is a modest one-refresh
# allowance: the totals of the launch night's first refresh imply an
# average factor near 0.75, and a many-fold player jump could move a rare
# trophy far more. Applied only while the history holds a single distinct
# state; a history that has already refreshed carries its own trend.
STALE_SHIFT = -0.30


def _clamp_rate(pct):
    return min(RATE_HI, max(RATE_LO, pct))


def _logit(pct):
    p = _clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hours_since_origin(text):
    return (_parse_time(text) - ORIGIN).total_seconds() / 3600.0


def _time_var(text):
    """Trend variable: ln(1 + hours since the release day began)."""
    return math.log1p(max(0.0, _hours_since_origin(text)))


def _round4(value):
    """Round to 4 decimals, half up, as a plain number."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _least_squares(xs, ys):
    """Return (intercept, slope, residual_std) of the OLS line, or None."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        resid = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
            / (n - 2)
        )
    else:
        resid = 0.0
    return intercept, slope, resid


def _distinct_states(history):
    """Collapse consecutive reads with identical players and rates."""
    states = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if not states or states[-1][0] != key:
            states.append((key, read))
    return [read for _key, read in states]


def _forecast_one(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target_x = _time_var(case["target"])

    if not history:
        return {"trophies": [
            {"id": t["id"], "q": _quantiles(_logit(1.0), _spread(horizon_days, 0.0, 0.0))}
            for t in trophies
        ]}

    checkpoint = _parse_time(case["at"])
    window_start = checkpoint.timestamp() - FIT_WINDOW_HOURS * 3600.0
    recent = [r for r in history
              if _parse_time(r["at"]).timestamp() >= window_start]
    if len(recent) < 3:
        recent = history
    last = history[-1]
    last_x = _time_var(last["at"])

    # Per-trophy trend fits and earner-count fits, then tier medians.
    fits = []
    for i, trophy in enumerate(trophies):
        xs, ys, es = [], [], []
        for read in recent:
            rate = read["rates"][i]
            if rate is None:
                continue
            players = read.get("players_tracked")
            if not players or players <= 0:
                continue
            xs.append(_time_var(read["at"]))
            ys.append(_logit(rate))
            earners = rate * players / 100.0
            es.append(math.log(earners) if earners > 0 else None)
        line = _least_squares(xs, ys)
        pairs = [(x, e) for x, e in zip(xs, es) if e is not None]
        eline = _least_squares([p[0] for p in pairs], [p[1] for p in pairs]) \
            if len(pairs) >= 3 else None
        fits.append({"line": line, "eline": eline, "xs": xs, "ys": ys})

    tier_slopes = {}
    tier_eslopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [f["line"][1] for f, t in zip(fits, trophies)
                  if t["tier"] == tier and f["line"] is not None]
        tier_slopes[tier] = _median(slopes) if slopes else 0.0
        eslopes = [f["eline"][1] for f, t in zip(fits, trophies)
                   if t["tier"] == tier and f["eline"] is not None]
        tier_eslopes[tier] = _median(eslopes) if eslopes else 0.0

    # Player-count growth (log-linear in the same time variable).
    pxs = [_time_var(r["at"]) for r in recent
           if r.get("players_tracked") and r["players_tracked"] > 0]
    pys = [math.log(r["players_tracked"]) for r in recent
           if r.get("players_tracked") and r["players_tracked"] > 0]
    pline = _least_squares(pxs, pys)
    players_last = last.get("players_tracked") or 0

    # Refresh-shock scatter: logit jumps between distinct reads.
    states = _distinct_states(recent)
    jump_sds = []
    for i in range(len(trophies)):
        jumps = []
        prev = None
        for read in states:
            rate = read["rates"][i]
            if rate is None:
                prev = None
                continue
            cur = _logit(rate)
            if prev is not None:
                jumps.append(cur - prev)
            prev = cur
        if len(jumps) >= 2:
            mean = sum(jumps) / len(jumps)
            var = sum((j - mean) ** 2 for j in jumps) / (len(jumps) - 1)
            jump_sds.append(math.sqrt(var))
        else:
            jump_sds.append(0.0)

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i]
        if last_rate is None:
            last_rate = 1.0
        fit = fits[i]

        # --- median model 1: shrunk logit trend ---
        if fit["line"] is not None:
            a, b, resid = fit["line"]
            b_shrunk = (OWN_SLOPE_WEIGHT * b
                        + (1.0 - OWN_SLOPE_WEIGHT) * tier_slopes[trophy["tier"]])
            trend_logit = a + b_shrunk * target_x
        else:
            trend_logit = _logit(last_rate)
            resid = 0.0

        # --- median model 2: earner counts over player counts ---
        count_logit = None
        if fit["eline"] is not None and players_last > 0:
            ea, eb, _er = fit["eline"]
            eb_shrunk = (OWN_SLOPE_WEIGHT * eb
                         + (1.0 - OWN_SLOPE_WEIGHT) * tier_eslopes[trophy["tier"]])
            last_earners = last_rate * players_last / 100.0
            if last_earners > 0:
                log_e_target = math.log(last_earners) + eb_shrunk * (target_x - last_x)
                if pline is not None:
                    log_p_target = (math.log(players_last)
                                    + pline[1] * (target_x - last_x))
                else:
                    log_p_target = math.log(players_last)
                rate_count = 100.0 * math.exp(log_e_target - log_p_target)
                count_logit = _logit(rate_count)

        if count_logit is None:
            median_logit = trend_logit
        else:
            median_logit = (TREND_WEIGHT * trend_logit
                            + (1.0 - TREND_WEIGHT) * count_logit)
        # Never-refreshed page: the next refresh dilutes every rate but
        # a universal one, so sit the median below the stale last rate.
        if len(states) <= 1 and last_rate < 99.99:
            median_logit += STALE_SHIFT
        median_logit = max(_logit(RATE_LO), min(_logit(RATE_HI), median_logit))

        # --- spread ---
        sigma = _spread(horizon_days, resid, jump_sds[i])
        rows.append({"id": trophy["id"],
                     "q": _quantiles(median_logit, sigma)})
    return {"trophies": rows}


def _spread(horizon_days, resid, jump_sd):
    walk = WALK_SIGMA * math.sqrt(horizon_days)
    floor = FLOOR_BASE + FLOOR_PER_SQRT_DAY * math.sqrt(horizon_days)
    total = math.sqrt(walk * walk + resid * resid
                      + jump_sd * jump_sd + floor * floor)
    return max(SPREAD_MIN, min(SPREAD_MAX, total))


def _quantiles(median_logit, sigma):
    out = []
    prev = None
    for z in Z_SCORES:
        q = _round4(_inv_logit(median_logit + z * sigma))
        q = min(100.0, max(0.0, q))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = _forecast_one(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
