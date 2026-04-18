# UCI Gas Sensor Benchmark Report

## Setup

- Dataset: official UCI Gas Sensor Array Drift at Different Concentrations, parsed from the ten batch files cached under `data/uci_gas_sensor_drift/`.
- Task family: binary one-vs-rest classification for analyte codes `1` through `5`.
- Split: train on batches `1-3`, validate on `4-5`, freeze the model, and evaluate deployment on `6-10`.
- Methods: `standard`, `isotropic` Jacobian penalty, and `DTR` with a drift basis estimated from unlabeled deployment covariate batch means.
- Selection rule: for each task and method family, choose the lambda with the best mean validation BCE across seeds.
- Note: the public sources disagree on the code-to-analyte name mapping, so the report keeps the official integer class codes instead of assigning gas names manually.

## Batch Totals

|   batch_id |   rows |
|-----------:|-------:|
|          1 |    445 |
|          2 |   1244 |
|          3 |   1586 |
|          4 |    161 |
|          5 |    197 |
|          6 |   2300 |
|          7 |   3613 |
|          8 |    294 |
|          9 |    470 |
|         10 |   3600 |

## Selected Lambdas

|   task_id | method    |   selected_lambda |   selection_val_bce |
|----------:|:----------|------------------:|--------------------:|
|         1 | dtr       |            3e-05  |            4.052    |
|         1 | isotropic |            0.0003 |            1.78296  |
|         1 | standard  |            0      |            9.21169  |
|         2 | dtr       |            0.0001 |            0.145063 |
|         2 | isotropic |            0.0003 |            0.024407 |
|         2 | standard  |            0      |            0.401215 |
|         3 | dtr       |            0.001  |            0.247389 |
|         3 | isotropic |            0.0003 |            0.151375 |
|         3 | standard  |            0      |            0.355168 |
|         4 | dtr       |            0.0003 |            6.80534  |
|         4 | isotropic |            0.001  |            2.38829  |
|         4 | standard  |            0      |            8.74498  |
|         5 | dtr       |            0.0001 |            2.41393  |
|         5 | isotropic |            0.001  |            0.9747   |
|         5 | standard  |            0      |            3.05674  |

## Overall Mean Results

| method    |   val_bce |   deploy_bce |   deploy_error |   volatility |   terminal_risk |   mean_gain |
|:----------|----------:|-------------:|---------------:|-------------:|----------------:|------------:|
| dtr       |    2.7327 |       1.4705 |         0.1252 |       1.6311 |          2.8635 |      1.7872 |
| isotropic |    1.0643 |       0.7917 |         0.1235 |       0.3575 |          1.4438 |      0.5625 |
| standard  |    4.354  |       1.7734 |         0.1328 |       2.0387 |          3.4349 |      2.0879 |

## Per-Task Selected Results

|   task_id | method    |   selected_lambda |   val_bce |   deploy_bce |   deploy_error |   volatility |   terminal_risk |   mean_gain |
|----------:|:----------|------------------:|----------:|-------------:|---------------:|-------------:|----------------:|------------:|
|         1 | dtr       |            3e-05  |    4.052  |       1.4143 |         0.2213 |       1.2377 |          3.1659 |      1.6819 |
|         1 | isotropic |            0.0003 |    1.783  |       0.7983 |         0.1831 |       0.4806 |          1.8664 |      0.6435 |
|         1 | standard  |            0      |    9.2117 |       1.7654 |         0.2454 |       1.8172 |          3.9399 |      1.5722 |
|         2 | dtr       |            0.0001 |    0.1451 |       0.3646 |         0.0832 |       0.0191 |          0.5147 |      1.1724 |
|         2 | isotropic |            0.0003 |    0.0244 |       0.3108 |         0.1056 |       0.007  |          0.4064 |      0.4291 |
|         2 | standard  |            0      |    0.4012 |       0.4483 |         0.0958 |       0.0265 |          0.5992 |      1.7364 |
|         3 | dtr       |            0.001  |    0.2474 |       1.1857 |         0.029  |       1.055  |          2.752  |      0.7911 |
|         3 | isotropic |            0.0003 |    0.1514 |       0.7092 |         0.0263 |       0.4295 |          1.7164 |      0.5209 |
|         3 | standard  |            0      |    0.3552 |       1.4917 |         0.0299 |       2.0061 |          3.7293 |      1.225  |
|         4 | dtr       |            0.0003 |    6.8053 |       2.9916 |         0.1952 |       2.0736 |          4.9356 |      2.3514 |
|         4 | isotropic |            0.001  |    2.3883 |       1.6114 |         0.2155 |       0.4461 |          2.2645 |      0.6688 |
|         4 | standard  |            0      |    8.745  |       3.714  |         0.2027 |       2.724  |          5.7153 |      3.1045 |
|         5 | dtr       |            0.0001 |    2.4139 |       1.3963 |         0.097  |       3.77   |          2.9493 |      2.9389 |
|         5 | isotropic |            0.001  |    0.9747 |       0.5289 |         0.0868 |       0.4244 |          0.9656 |      0.5505 |
|         5 | standard  |            0      |    3.0567 |       1.4475 |         0.0902 |       3.62   |          3.1908 |      2.8012 |

## Win Counts

- DTR vs standard: volatility wins `4/5`, terminal-risk wins `5/5`, deploy-BCE wins `5/5`, deploy-error wins `4/5`.
- DTR vs isotropic: volatility wins `0/5`, terminal-risk wins `0/5`, deploy-BCE wins `0/5`, deploy-error wins `2/5`.

## Quick Read

- Read `deploy_bce` and `deploy_error` as the main deployment-quality metrics.
- Read `volatility` as the temporal instability metric over deployment batches `6-10`.
- Read `mean_gain` as the average directional tangent gain along the realized batch-to-batch covariate motion.

## Findings

- The second benchmark does support the weaker claim that Jacobian regularization helps under drift: relative to the unregularized model, DTR wins on `5/5` tasks in deployment BCE and terminal risk, and on `4/5` tasks in volatility and deployment error.
- The stronger directional claim does not replicate here. Isotropic regularization beats DTR on `5/5` tasks in deployment BCE, `5/5` in terminal risk, and `5/5` in volatility; DTR only wins on deployment error for `2/5` tasks.
- Overall means make the pattern clear: `standard` deploy BCE `1.7734`, `DTR` deploy BCE `1.4705`, `isotropic` deploy BCE `0.7917`. The same ordering holds for volatility and terminal risk.
- The best reading of this benchmark is therefore mixed: the gas-sensor dataset supports "regularization helps under drift" but does not support "drift-aligned DTR is better than isotropic smoothing" under this protocol.

## Follow-up 1D Basis Check

- I also ran a follow-up DTR-only sweep with a one-dimensional drift basis to check whether the main result was being hurt by an over-wide estimated subspace.
- That follow-up did not change the conclusion. The selected 1D DTR models had overall mean validation BCE `1.8226`, deployment BCE `1.5479`, deployment error `0.1217`, volatility `1.6738`, terminal risk `2.8770`, and mean gain `1.3729`.
- So a narrower drift basis improved some DTR validation numbers, but it still did not approach the isotropic benchmark on deployment BCE or volatility. The gap is not just a `2D` versus `1D` basis issue.

## Diagnostic Interpretation

- The main reason this benchmark resists the paper's directional story is that the one-vs-rest classification protocol is not close to pure covariate drift. Batch composition changes heavily over time, so unlabeled pooled batch means mix sensor drift with changing class proportions.
- The batch priors move a lot. For example, class `5` appears with rates `0.157`, `0.428`, `0.173`, `0.075`, `0.320`, `0.263`, `0.174`, `0.486`, `0.166`, `0.167` across batches `1` through `10`, and class `4` swings from `0.068` to `0.234` to `0.013` to `0.206`.
- The pooled deployment mean shifts are still low-rank, with the top singular direction explaining about `81.4%` of the variance, but that low-rank structure is not enough by itself because it is contaminated by changing gas composition.
- I checked that directly with an oracle basis built from positive-class deployment means for each one-vs-rest task. Even then, DTR still did not catch isotropic smoothing on deployment BCE or volatility.
- The oracle basis did improve discrimination-oriented metrics somewhat. Overall oracle-DTR deployment AUC was `0.9174` and deployment error was `0.1119`, versus isotropic deployment AUC `0.9251` and deployment error `0.1235`. So the remaining gap is more about probability-quality and stability metrics than about pure binary accuracy alone.
- That pattern is consistent with the idea that isotropic smoothing is a stronger bias when the benchmark combines sensor drift with class-prior shift and likely task-composition shift. DTR helps relative to standard training, but the mixed-shift protocol is outside the paper's clean low-rank covariate-drift regime.

## Exploratory Within-Gas Regression Probe

- To check whether the benchmark becomes more favorable when the label semantics are fixed, I ran a separate exploratory probe on the same UCI dataset using within-gas concentration regression instead of one-vs-rest classification.
- In that probe, each gas class is treated as its own regression task: train on batches `1-3`, validate on `4-5`, freeze the regressor, and evaluate deployment on `6-10`. I used the same `standard`, `isotropic`, and `DTR` comparison, with a one-dimensional drift basis because that was the best directional variant in the classification diagnosis.
- The overall averages across gas classes `1-5` were much closer than in the one-vs-rest setting. Mean deployment MSE was `1.7363` for `standard`, `1.6568` for `isotropic`, and `1.6446` for `DTR`.
- The per-gas pattern was mixed rather than uniformly directional:
  - Gas `1`: DTR was clearly best on deployment MSE (`1.1657` versus `1.3770` isotropic and `1.6996` standard) and on volatility (`0.3426` versus `0.6107` isotropic and `0.5112` standard).
  - Gas `2`: isotropic was better than DTR on deployment MSE (`0.8084` versus `0.9324`) and volatility (`0.7519` versus `1.0946`).
  - Gases `3` and `4`: the methods were nearly tied.
  - Gas `5`: isotropic was slightly better, but the gap was much smaller than in the one-vs-rest classification benchmark.
- That exploratory result is the cleanest evidence so far that the harsh one-vs-rest benchmark is part of the problem. Once the task is reformulated so the label semantics stay fixed, DTR becomes competitive again and can win clearly on some gases.
