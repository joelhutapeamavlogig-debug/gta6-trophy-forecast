# GTA VI Trophy Rarity Forecaster

Entry for make.fun bounty [`gta6-trophy-rarity-forecast-01`](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast each Grand Theft Auto VI (PS5) trophy's Exophase earn rate at the target instant,
as five quantiles per trophy, scored by mean pinball loss on the logit scale.

## Model

The program is **refresh-aware** and **tier-aware**, built around the identity
`earned_count = rate × players_tracked / 100`:

1. **Refresh detection** — Exophase refreshes a game's counts on its own schedule; the player
   base jumps from the launch-night baseline (638) to the tracked population. All reads before
   the first >3× player-count jump are dropped, so the fit uses only post-refresh data.
2. **Earners/players decomposition** — earned count is fitted as
   `ln(ec) = a·ln(1 + hours_since_launch) + b` by least squares over the surviving reads;
   the player count is projected with the same log-linear form. The forecast rate at the
   target is `eĉ(target) / playerŝ(target) × 100`.
3. **Tier growth prior** — when a trophy's history is flat (no signal), a gentle
   tier-dependent growth factor is applied (platinum 0.010 → bronze 0.060 h⁻¹):
   early-game bronze trophies climb faster than endgame platinum ones.
4. **Spread** — quadrature combination of the house random-walk spread
   (`0.5·√days` on the logit scale) and the regression residual; never tighter than the
   baseline. Quantiles are `sigmoid(logit(median) + z_τ·spread)` at
   τ ∈ {0.05, 0.25, 0.50, 0.75, 0.95}.
5. **Degenerate guards** — rates are clamped to [0.01, 99.99] before every logit, quantiles
   are forced non-decreasing, and a trophy with no usable history falls back to the house
   baseline formula.

## Constraints honored

- Python 3.13, standard library only, offline, deterministic (no randomness).
- ≤ 8 s, 1 CPU, 1024 MB per case; reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`),
  writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Run locally

```bash
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see [LICENSE](LICENSE).
