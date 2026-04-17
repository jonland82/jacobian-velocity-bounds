# Tetouan Power Consumption Benchmark

- Dataset: `tetouan_zone1_power`.
- Target: `Zone 1 Power Consumption`.
- Train rows: `17280`.
- Validation rows: `8784`.
- Deployment rows: `26352` across `6` blocks.
- Chronological split: train before `2017-05-01`, validate until `2017-07-01`, freeze, then deploy.
- Standard deploy MSE: `1.161e+08`.
- Isotropic deploy MSE: `9.314e+07`.
- DTR deploy MSE: `6.838e+07`.
- Standard volatility: `1.092e+16`.
- Isotropic volatility: `4.887e+15`.
- DTR volatility: `1.769e+15`.

Takeaway: overall deploy MSE winner `dtr`; overall volatility winner `dtr`; DTR vs isotropic deploy MSE favored `dtr`; DTR vs isotropic volatility favored `dtr`.
