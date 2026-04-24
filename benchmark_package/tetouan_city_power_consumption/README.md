# Tetouan Power Consumption Benchmark

- Dataset: `tetouan_zone1_power`.
- Target: `Zone 1 Power Consumption`.
- Train rows: `17280`.
- Validation rows: `8784`.
- Deployment rows: `26352` across `6` blocks.
- Chronological split: train before `2017-05-01`, validate until `2017-07-01`, freeze, then deploy.
- Matched seeds: `10`.
- Selection rule: per-method lambda chosen by validation MSE, with validation directional gain used only as a secondary tie-breaker.
- Validation-selected isotropic lambda: `3.000e-04`.
- Validation-selected DTR lambda: `1.000e-02`.
- Standard deploy MSE: `1.077e+08 +/- 1.114e+08`.
- Isotropic deploy MSE: `1.007e+08 +/- 8.075e+07`.
- DTR deploy MSE: `6.824e+07 +/- 4.868e+07`.
- Standard volatility: `1.069e+16 +/- 1.657e+16`.
- Isotropic volatility: `7.198e+15 +/- 1.024e+16`.
- DTR volatility: `3.071e+15 +/- 4.321e+15`.
- Paired DTR-vs-standard volatility wins: `8 / 10`.
- Paired DTR-vs-isotropic volatility wins: `8 / 10`.

Takeaway: validation-selected DTR has the best mean deploy MSE and volatility on Tetouan, with paired seed volatility improvements over both baselines.
