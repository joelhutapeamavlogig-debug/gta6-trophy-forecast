#!/usr/bin/env python3
"""
Premium GTA VI Trophy Rarity Forecaster — gta6-trophy-rarity-forecast-01

Philosophy:
- Top entries are "refresh-aware earners/players" forecasters.
- Core model: earned_count = rate * players_tracked / 100.
  Track earned_count growth, project forward, divide by projected players.
- Tier-aware growth priors: common trophies grow fast, rare/endgame slow.
- When history shows no signal (all identical rates), fall back to baseline.
- Spread: baseline random walk + optional fit residual.
"""

import math, json, os, calendar
from typing import Optional

INPUT_DIR  = os.environ.get("MAKEFUN_INPUT_DIR",  "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS     = [0.05, 0.25, 0.5, 0.75, 0.95]
Z          = [-1.645, -0.674, 0.0, 0.674, 1.645]
BASE_SPREAD = 0.5   # logit / sqrt-day (house baseline)
LAUNCH_TS   = 1731964800.0   # 2026-11-19T00:00:00Z


# ── helpers ──────────────────────────────────────────────────────────────────

def clamp(p: float) -> float:
    return max(0.01, min(99.99, p))

def logit(p: float) -> float:
    return math.log(clamp(p) / (100.0 - clamp(p)))

def sigmoid(x: float) -> float:
    ex = math.exp(-x)
    return 100.0 / (1.0 + ex)

def r4(x: float) -> float:
    return round(x * 10_000) / 10_000

def iso_to_ts(s: str) -> float:
    s = s.replace("Z", "").replace("T", "-").replace(":", "-")
    parts = s.split("-")
    y = int(parts[0]); mo = int(parts[1]); d = int(parts[2])
    h = int(parts[3]); mi = int(parts[4])
    sec = int(parts[5].split(".")[0])
    return calendar.timegm((y, mo, d, h, mi, sec, 0, 0, 0))


# ── tier growth priors ───────────────────────────────────────────────────────

# k: hours^{-1} for logistic growth ceiling tracking
# Platinum/gold = slow, silver/bronze = faster
# Represents how fast players earn each category (early-game bronze = fast)
TIER_K = {
    "platinum": 0.010,
    "gold":     0.020,
    "silver":   0.035,
    "bronze":   0.060,
}
DEFAULT_K = 0.025


# ── core forecasting ──────────────────────────────────────────────────────────

def detect_refresh(history: list) -> int:
    """
    Returns the index of first read after the player-count refresh.
    Before refresh: all reads show 638 players (house baseline).
    Refresh: player count jumps >3x.
    """
    if len(history) < 2:
        return 0
    players_seq = [r["players_tracked"] for r in history]
    base = players_seq[0]
    for i, p in enumerate(players_seq):
        if p > 3 * base:
            return i
    return 0   # no clear refresh detected


def build_observations(history: list, tid: int):
    """
    Returns list of {h: hours_from_launch, ec: earned_count, r: rate, players: n}.
    Filters to non-null rates.  Uses post-refresh data when available.
    """
    rid = tid - 1
    ref_idx = detect_refresh(history)

    obs = []
    for read in history:
        rates = read.get("rates") or []
        if rid < len(rates) and rates[rid] is not None:
            r = rates[rid]
            players = read.get("players_tracked", 0)
            ec = r * players / 100.0
            h = (iso_to_ts(read["at"]) - LAUNCH_TS) / 3600.0
            obs.append({"h": h, "r": r, "ec": ec, "players": players})

    if not obs:
        return [], 0

    # Post-refresh subset
    post_refs = obs[ref_idx:] if ref_idx > 0 else obs
    return obs, ref_idx


def fit_logit_growth(obs: list, target_hours: float) -> tuple[float, float, float]:
    """
    Fit ln(ec) = a*ln(1+h) + b  (ec = earned count).
    Returns (a, b, residual_std).  Uses all obs.
    """
    pts = [(math.log1p(o["h"]), math.log(max(o["ec"], 1e-6))) for o in obs]
    n = len(pts)
    if n < 2:
        return 0.0, math.log(max(obs[-1]["ec"], 1e-6)), 0.0
    t_v = [p[0] for p in pts]; l_v = [p[1] for p in pts]
    t_m = sum(t_v) / n; l_m = sum(l_v) / n
    num = sum((t_v[i] - t_m) * (l_v[i] - l_m) for i in range(n))
    den = sum((t_v[i] - t_m) ** 2 for i in range(n))
    a = num / den if den > 1e-8 else 0.0
    b = l_m - a * t_m
    residuals = [l_v[i] - (a * t_v[i] + b) for i in range(n)]
    ss = sum(r * r for r in residuals)
    std = math.sqrt(ss / max(n - 2, 1)) if n > 2 else math.sqrt(ss / n if n > 0 else 0.0)
    return a, b, std


def forecast_trophy(
    tid: int,
    tier: str,
    history: list,
    target_hours: float,
    horizon_days: float,
) -> list[float]:
    """
    Returns [q05, q25, q50, q75, q95] for one trophy.
    """
    obs, ref_idx = build_observations(history, tid)
    if not obs:
        return baseline_quantiles(0.01, horizon_days)

    last_obs = obs[-1]
    last_r = last_obs["r"]
    last_ec = last_obs["ec"]
    last_players = last_obs["players"]

    # Detect if all rates are identical (flat history)
    rates_all_same = len(set(o["r"] for o in obs)) == 1

    # ── If rates flat → dilution-aware stale-page model ───────────────────
    # Exophase refreshes counts on its own schedule: between refreshes every
    # read is identical. The house's launch-night recording (RULES) showed the
    # page unchanged at 638 players through 04:00Z and refreshed by 04:30Z at
    # 3,868 players — a tracked-player dilution of 3,868/638 = 6.06 in the
    # first ~4.7 hours, with every rate but the automatic 100% one dropping.
    # So when the page is stale, the rates at the target reflect the NEXT
    # refresh's (larger) tracked base: forecast the dilution, not growth.
    if rates_all_same:
        # The automatic 100% trophy never dilutes (every tracked player has it)
        if last_r >= 99.95:
            return baseline_quantiles(100.0, horizon_days)

        if last_players < 2000:
            # Early/stale tracking: the next refresh brings the tracked base to
            # launch-week scale (~6.06x per the house's published reading).
            # Earners also keep accruing while the page is stale: bump by a
            # modest launch-hour accrual, capped so long horizons stay sane.
            horizon_hours = horizon_days * 24.0
            bump = min(0.07 * horizon_hours, 0.60)
            dilution = 3868.0 / 638.0
            proj_r = last_r * (1.0 + bump) / dilution
            # Honest hedge: the refresh might land late (rates stay) — widen.
            spread = 0.30
        else:
            # Stale at launch-week scale: a further refresh dilutes by the
            # (unknown) adoption since — hedge wide around a mild dilution.
            proj_r = last_r / 1.8
            spread = 1.1
        median_l = logit(proj_r)
        qs = []
        for z in Z:
            q_l = max(logit(0.01), min(logit(99.99), median_l + z * spread))
            qs.append(r4(sigmoid(q_l)))
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        return [r4(q) for q in qs]

    # ── Logit-linear fit on earned count ─────────────────────────────────
    a, b, res_std = fit_logit_growth(obs, target_hours)

    # Project earned count at target
    t_target = math.log1p(target_hours)
    proj_ln_ec = a * t_target + b
    proj_ec = math.exp(proj_ln_ec)

    # Project players at target
    # Players grow from pre-refresh baseline to post-refresh level.
    # Use simple exponential model for player growth.
    if ref_idx > 0 and len(obs) >= ref_idx:
        post_obs = obs[ref_idx:]
        if len(post_obs) >= 2:
            # Player count exponential growth
            p0 = post_obs[0]["players"]
            p1 = post_obs[-1]["players"]
            h0 = post_obs[0]["h"]
            h1 = post_obs[-1]["h"]
            if p1 > p0 and h1 > h0:
                pg_a = math.log(p1 / p0) / math.log1p(h1)
                pg_b = math.log(p0) - pg_a * math.log1p(h0)
                proj_players = math.exp(pg_a * t_target + pg_b)
            else:
                proj_players = last_players
        else:
            proj_players = last_players
    else:
        # No refresh: assume constant players
        proj_players = last_players

    proj_players = max(proj_players, last_players * 0.5)  # floor at 50% of last

    # Forecast rate = projected earned_count / projected players
    proj_rate = (proj_ec / proj_players) * 100.0 if proj_players > 0 else last_r
    proj_rate = clamp(proj_rate)

    # ── Spread ──────────────────────────────────────────────────────────────
    rw_spread = BASE_SPREAD * math.sqrt(horizon_days)
    total_std = math.sqrt(rw_spread ** 2 + res_std ** 2) if res_std > 0.01 else rw_spread
    spread = max(total_std, rw_spread)

    # ── Handle extremes ────────────────────────────────────────────────────
    if proj_rate >= 99.5:
        proj_rate = 99.5  # cap

    # Build quantiles
    median_l = logit(proj_rate)
    qs = []
    for z in Z:
        q_l = median_l + z * spread
        q_l = max(logit(0.01), min(logit(99.99), q_l))
        qs.append(r4(sigmoid(q_l)))

    # Non-decreasing
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return [r4(q) for q in qs]


def baseline_quantiles(rate: float, horizon_days: float) -> list[float]:
    """House baseline: rate stays, spread widens by random walk."""
    r = clamp(rate)
    l = logit(r)
    spread = BASE_SPREAD * math.sqrt(horizon_days)
    qs = []
    for z in Z:
        q_l = l + z * spread
        q_l = max(logit(0.01), min(logit(99.99), q_l))
        qs.append(r4(sigmoid(q_l)))
    return qs


# ── main ─────────────────────────────────────────────────────────────────────

def process_case(case: dict) -> dict:
    trophies = case["trophies"]
    history  = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target_ts    = iso_to_ts(case["target"])
    target_hours = (target_ts - LAUNCH_TS) / 3600.0

    rows = []
    for t in trophies:
        tid  = t["id"]
        tier = t.get("tier", "bronze")
        qs = forecast_trophy(tid, tier, history, target_hours, horizon_days)
        rows.append({"id": tid, "q": qs})

    return {"trophies": rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    print(f"Cases: {files}")

    for fname in files:
        case_id = fname[:-3]
        with open(os.path.join(INPUT_DIR, fname)) as f:
            case = json.load(f)

        output = process_case(case)
        out_path = os.path.join(OUTPUT_DIR, f"{case_id}.out")
        with open(out_path, "w") as f:
            json.dump(output, f)

        print(f"Wrote {out_path}: {len(output['trophies'])} trophies")
        for t in output["trophies"][:5]:
            print(f"  id={t['id']:2d} q={[round(q,4) for q in t['q']]}")


if __name__ == "__main__":
    main()
