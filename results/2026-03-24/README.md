# 2026-03-24 Evaluation Artifacts

This folder contains the final comparison artifacts exported from the long headless training run on 2026-03-24.

Files:
- `baseline_straight.mp4`: straight driving without the learned suspension policy
- `learned_straight.mp4`: straight driving with the learned suspension policy
- `tilt_metrics.csv`: roll/pitch RMS, peak tilt, and forward velocity summary

Source checkpoint:
- `logs/rsl_rl/inventory_leveling/2026-03-24_05-32-50_wallclock_5h_chunk_36_2026-03-24_05-32-42/model_5250.pt`

Notes:
- The learned policy reduced pitch RMS compared with baseline.
- Peak tilt and forward speed still need tuning, so this should be treated as the current milestone result rather than the final target result.
