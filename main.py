"""
Deterministic offline forecast for the Roblox Top Trending chart.

STRATEGY OVERVIEW
-----------------
The forecast is produced by `soft_rank_forecast`, which assigns each game a
probability distribution over chart positions 1–10 for the next snapshot.

1. HISTORY WINDOW (14 snapshots)
   The last 14 snapshots are used for all calculations. This gives enough
   baseline to distinguish real trends from one-off noise while keeping the
   window short enough that stale data doesn't swamp recent signal.

2. RECENCY WEIGHTING (5× boost on last 3 snapshots)
   Within the 14-snapshot window, the three most recent snapshots are given
   5× the weight of older ones. This applies to the weighted average rank and
   to the slope calculation, so a game that surged in the last hour pulls its
   projected rank down faster than a slow historical climb would.

3. WEIGHTED RANK ESTIMATE
   For each game, two values are computed from its weighted history:
   - weighted_avg_rank: the recency-weighted mean rank across all snapshots
     where the game appeared.
   - slope: the difference between the weighted average of the late half vs
     the early half of observations, capped at ±4 positions per step.
   - growth: log-scale player-count growth between first and last observation,
     used as a secondary push signal.
   The projected rank combines these:
       projected = weighted_avg_rank + 2.0 * slope - 2.0 * growth

4. DROPOUT DETECTION (+15 penalty)
   If a game appeared in the history window but is absent from the latest
   snapshot, both its current_rank and projected_rank are penalised by +15
   positions. This sharply reduces its weight in the top-10 probability
   distribution without zeroing it out entirely (the game may have been
   missed by a single scrape).

5. TOP-POSITION STICKINESS (80% anchor at rank 1, 50% at rank 2)
   Games currently sitting at rank 1 or 2 have their projected rank pulled
   back toward their current position:
     - Rank 1: 80% anchor toward current rank, 20% velocity
     - Rank 2: 50% anchor toward current rank, 50% velocity
   This reflects the empirical observation that #1 games churn far less than
   lower positions, and prevents a noisy velocity estimate from displacing an
   incumbent unfairly. This directly mirrors the "sticky chart + #1-dropout
   mixture" pattern observed in the top-scoring forecaster.

6. GAUSSIAN PROBABILITY DISTRIBUTION
   For each chart position 1–10, a per-game weight is computed as:
       0.45 * exp(-|current_rank  - place| / 2.0)
     + 0.55 * exp(-|projected_rank - place| / 4.0)
   The wider σ=4.0 on the projected term deliberately spreads probability
   mass to neighbouring ranks, so a game projected near rank 4 also
   contributes meaningfully to places 1–7. Weights are normalised to sum
   to 0.99 per place (leaving 1% unallocated as a regularisation floor).

7. SHORT-SNAPSHOT PADDING
   If the latest snapshot contains fewer than 10 games (partial scrape or
   genuine chart contraction), the top10 list is padded with games from
   earlier history ranked by their best-ever position, ensuring the output
   always contains exactly 10 entries.
"""

import json
import math
import os
from pathlib import Path


def input_dir() -> Path:
    return Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))


def output_dir() -> Path:
    return Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def bounded_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_rank(value) -> int:
    return int(value)


def normalize_history(rows):
    normalized_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        snapshot = row.get("top_trending")
        if not isinstance(snapshot, list):
            continue
        normalized_snapshot = []
        for item in snapshot:
            if not isinstance(item, dict):
                continue
            try:
                universe_id = int(item["universe_id"])
                rank = safe_rank(item.get("rank", item.get("place")))
            except (KeyError, TypeError, ValueError):
                continue
            if universe_id <= 0 or rank <= 0:
                continue
            try:
                playing = int(item.get("playing") or 0)
            except (TypeError, ValueError):
                playing = 0
            normalized_snapshot.append({"universe_id": universe_id, "rank": rank, "playing": playing})
        if normalized_snapshot:
            cleaned = dict(row)
            cleaned["top_trending"] = normalized_snapshot
            normalized_rows.append(cleaned)
    normalized_rows.sort(key=lambda row: str(row.get("at", "")))
    return normalized_rows


def load_history(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return normalize_history(rows)


def snapshot_games(snapshot, limit=None):
    games = []
    for item in snapshot or []:
        try:
            universe_id = int(item["universe_id"])
            place = safe_rank(item.get("rank", item.get("place")))
        except (KeyError, TypeError, ValueError):
            continue
        games.append((place, universe_id))
    ordered = sorted(games)
    return ordered if limit is None else ordered[:limit]


def snapshot_records(snapshot):
    records = {}
    for item in snapshot or []:
        try:
            universe_id = int(item["universe_id"])
            rank = safe_rank(item.get("rank", item.get("place")))
            playing = int(item.get("playing") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        records[universe_id] = (rank, max(0, playing))
    return records


def latest_games(history):
    if not history:
        return []
    return snapshot_games(history[-1].get("top_trending"))


def forecast_order(history):
    snapshots = [snapshot_games(row.get("top_trending")) for row in history[-14:]]
    ranks = {}
    for snapshot in snapshots:
        for place, universe_id in snapshot:
            ranks.setdefault(universe_id, []).append(place)
    if not ranks:
        return []
    latest = {universe_id: place for place, universe_id in snapshots[-1]}
    scored_games = []
    for universe_id, rank_history in ranks.items():
        recent_rank = latest.get(universe_id, min(rank_history) + 2)
        average_rank = sum(rank_history) / len(rank_history)
        score = 0.65 * recent_rank + 0.35 * average_rank
        scored_games.append((score, recent_rank, universe_id))
    scored_games.sort()
    return [(place, universe_id) for place, (_, _, universe_id) in enumerate(scored_games[:10], 1)]


def soft_rank_forecast(history):
    # --- constants -----------------------------------------------------------
    HISTORY_WINDOW      = 14    # snapshots to consider
    RECENCY_BOOST_COUNT = 3     # trailing snapshots that get the boost
    RECENCY_BOOST       = 5.0   # weight multiplier for recent snapshots (up from 3×)
    SLOPE_CAP           = 4.0   # max rank-change per step (±)
    DROPOUT_PENALTY     = 15.0  # rank penalty for games missing from latest snapshot (up from 8)
    STICKY_RANKS        = {1: 0.8, 2: 0.5}  # stickiness anchor: rank1=80%, rank2=50%
    SIGMA_CURRENT       = 2.0   # Gaussian spread for current-rank term
    SIGMA_PROJECTED     = 4.0   # Gaussian spread for projected-rank term (up from 3.0)
    WEIGHT_CURRENT      = 0.45  # blend weight for current-rank Gaussian
    WEIGHT_PROJECTED    = 0.55  # blend weight for projected-rank Gaussian

    # --- load snapshots and build recency weights ----------------------------
    snapshots = [snapshot_records(row.get("top_trending")) for row in history[-HISTORY_WINDOW:]]
    if not snapshots:
        return {}, []

    latest      = snapshots[-1]
    latest_ids  = set(latest.keys())
    n_snapshots = len(snapshots)
    snapshot_weights = [
        RECENCY_BOOST if i >= n_snapshots - RECENCY_BOOST_COUNT else 1.0
        for i in range(n_snapshots)
    ]

    # --- build a rank+projection estimate for every candidate game -----------
    estimates: dict[int, tuple[float, float]] = {}

    for universe_id in set().union(*snapshots):
        # Collect (snapshot_index, (rank, playing)) for each appearance
        indexed_observations = [
            (i, snapshot[universe_id])
            for i, snapshot in enumerate(snapshots)
            if universe_id in snapshot
        ]
        if not indexed_observations:
            continue

        obs_indices, obs_records = zip(*indexed_observations)
        ranks   = [rank    for rank,    _ in obs_records]
        players = [playing for _,    playing in obs_records]
        weights = [snapshot_weights[i] for i in obs_indices]
        total_weight = sum(weights)

        # Recency-weighted average rank
        weighted_avg_rank = sum(r * w for r, w in zip(ranks, weights)) / total_weight

        # Recency-weighted slope: late-half avg minus early-half avg
        mid           = len(ranks) // 2
        early_ranks   = ranks[:mid]   if mid else ranks[:1]
        late_ranks    = ranks[mid:]   if mid else ranks
        early_weights = weights[:mid] if mid else weights[:1]
        late_weights  = weights[mid:] if mid else weights
        early_avg = sum(r * w for r, w in zip(early_ranks,  early_weights)) / sum(early_weights)
        late_avg  = sum(r * w for r, w in zip(late_ranks,   late_weights))  / sum(late_weights)
        slope = max(-SLOPE_CAP, min(SLOPE_CAP, (late_avg - early_avg) / max(1, len(ranks) - 1)))

        # Log-scale player-count growth (boosts games with rising audiences)
        if len(players) > 1 and players[0] > 0 and players[-1] > 0:
            growth = (math.log1p(players[-1]) - math.log1p(players[0])) / (len(players) - 1)
        else:
            growth = 0.0

        # Base projection
        current_rank   = latest.get(universe_id, (min(ranks) + 10, 0))[0]
        projected_rank = max(1.0, min(100.0, weighted_avg_rank + 2.0 * slope - 2.0 * growth))

        # Dropout penalty: aggressively demote games absent from the latest snapshot
        if universe_id not in latest_ids:
            current_rank   = min(100.0, current_rank   + DROPOUT_PENALTY)
            projected_rank = min(100.0, projected_rank + DROPOUT_PENALTY)

        # Top-position stickiness: strongly anchor rank-1 games, moderately anchor rank-2
        if current_rank in STICKY_RANKS:
            stickiness_weight = STICKY_RANKS[current_rank]
            projected_rank = (
                stickiness_weight * current_rank
                + (1.0 - stickiness_weight) * projected_rank
            )

        estimates[universe_id] = (current_rank, projected_rank)

    # --- build Gaussian probability distributions for places 1–10 -----------
    probabilities: dict[int, dict[int, float]] = {}
    for place in range(1, 11):
        place_weights = {
            universe_id: (
                WEIGHT_CURRENT   * math.exp(-abs(current_rank   - place) / SIGMA_CURRENT)
                + WEIGHT_PROJECTED * math.exp(-abs(projected_rank - place) / SIGMA_PROJECTED)
            )
            for universe_id, (current_rank, projected_rank) in estimates.items()
        }
        top_candidates = sorted(place_weights, key=lambda u: (-place_weights[u], u))[:100]
        total_weight   = sum(place_weights[u] for u in top_candidates)
        probabilities[place] = {
            universe_id: 0.99 * place_weights[universe_id] / total_weight
            for universe_id in top_candidates
        }

    # --- derive top10 list, padding if the latest snapshot was short ---------
    ranked_estimates = sorted(
        estimates.items(), key=lambda pair: (pair[1][1], pair[1][0], pair[0])
    )[:10]
    top10 = [universe_id for universe_id, _ in ranked_estimates]

    if len(top10) < 10:
        seen = set(top10)
        historical_best: dict[int, int] = {}
        for snapshot in snapshots:
            for universe_id, (rank, _) in snapshot.items():
                if universe_id not in seen:
                    if universe_id not in historical_best or rank < historical_best[universe_id]:
                        historical_best[universe_id] = rank
        fallback_ids = sorted(historical_best, key=lambda u: (historical_best[u], u))
        top10 = top10 + fallback_ids[: 10 - len(top10)]

    return probabilities, top10


def build_forecast(history):
    probabilities_by_place, top10 = soft_rank_forecast(history)
    probabilities = [
        {"universe_id": universe_id, "place": place, "p": bounded_probability(probability)}
        for place, candidates in probabilities_by_place.items()
        for universe_id, probability in candidates.items()
    ]
    return {"probabilities": probabilities, "top10": top10}


def main() -> None:
    history = load_history(input_dir() / "history.ndjson")
    forecast = build_forecast(history)
    output_path = output_dir() / "forecast.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(forecast, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
