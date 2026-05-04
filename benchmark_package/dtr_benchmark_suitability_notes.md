# DTR Benchmark Suitability Notes

This note records what we learned from the real-data benchmark search around drift-aligned tangent regularization (DTR). The goal is not only to remember which datasets were kept or dropped, but to make explicit **when DTR is the right inductive bias and when it is not**.

Some discarded benchmark branches were intentionally removed from the cleaned repo. Where exact numbers are included below, they come either from retained diagnostic reports or from exploratory run summaries recorded during the benchmark search. These notes are benchmark-search context, not additional manuscript experiments.

## 1. The regime DTR is designed for

Across deployment blocks indexed by $t$, let

$$
r_t(\theta) = \mathbb{E}[\ell(f_\theta(X_t), Y_t)]
$$

denote deployment risk, and let $V \in \mathbb{R}^{d \times k}$ span the dominant drift subspace estimated from deployment covariates. DTR trains with

$$
\mathcal{L}_{\mathrm{DTR}}(\theta)
=
\mathbb{E}[\ell(f_\theta(X),Y)]
\;+\;
\lambda \,\mathbb{E}\|J_f(X)V\|_F^2.
$$

The low-rank logic behind DTR is strongest when all of the following are approximately true:

1. The deployment covariates move mainly in a small subspace $V$, so the realized drift is close to low rank.
2. The target semantics stay fixed over time, so changes in deployment blocks are mostly changes in $X_t$, not changes in what the label means.
3. The variation in $r_t$ is materially driven by tangent amplification along the realized covariate motion.

Equivalently, DTR is a good fit when the important part of the deployment path looks like

$$
X_t \approx X_0 + V z_t + \varepsilon_t,
\qquad
\|\varepsilon_t\| \text{ small relative to } \|Vz_t\|.
$$

When that picture breaks, isotropic smoothing or even plain ERM can be as good as or better than DTR. The main failure modes we saw were:

- label-mix or task-composition shift contaminating unlabeled mean-shift estimates,
- high-rank or multi-regime drift, so a small fixed $V$ misses too much residual motion,
- latent hidden-state dynamics that are not well expressed in the observed covariates.

## 2. Benchmarks we tried and discarded

### 2.1 UCI Gas Sensor Array Drift, one-vs-rest classification

This was the clearest "do not use DTR here" case.

High-level outcome from the retained report:

- Overall deployment BCE: `standard` `1.7734`, `DTR` `1.4705`, `isotropic` `0.7917`.
- DTR beat `standard` on `5/5` tasks in deployment BCE and terminal risk.
- Isotropic beat DTR on `5/5` tasks in deployment BCE, `5/5` in volatility, and `5/5` in terminal risk.

Why this benchmark was not well suited:

For one-vs-rest classification, pooled batch means satisfy

$$
\mu_t = \sum_{y \in \{0,1\}} p_t(y)\,\mu_{t,y}.
$$

So the unlabeled mean shift

$$
\mu_t - \mu_{t-1}
$$

changes not only when sensor responses drift, but also when the class proportions $p_t(y)$ change. In this protocol, batch composition changed heavily over time. That means the estimated drift directions mixed together:

- actual sensor/covariate drift,
- changing class priors,
- broader task-composition shift.

That is outside the clean covariate-drift regime DTR is meant to exploit. The low-rank directional penalty was then trying to regularize against a confounded object.

The diagnostics reinforced that interpretation:

- A narrower `1D` DTR basis did not rescue deployment BCE or volatility.
- An oracle class-conditional basis improved some discrimination-style metrics, but still did not catch isotropic smoothing on the paper's main deployment-risk metrics.
- An exploratory within-gas **regression** reformulation was much closer: mean deployment MSE across gases `1-5` was `1.7363` for `standard`, `1.6568` for `isotropic`, and `1.6446` for `DTR`.

So the main lesson from this dataset is precise:

> DTR can help under drift, but it is a poor match for benchmarks where unlabeled block summaries are strongly contaminated by changing label mix.

Reference: [`uci_gas_sensor_benchmark_report.md`](./uci_gas_sensor_benchmark_report.md)

### 2.2 Beijing Multi-Site Air Quality

This benchmark produced a mixed result rather than a directional win:

- In the earlier exploratory run, `standard` had the best deployment MSE.
- `isotropic` had the best volatility.
- DTR was not the overall winner on the main risk metrics.

Why it was not a clean DTR benchmark:

The dataset combines multi-year pollution dynamics, site effects, weather, and broader urban regime changes. That makes the effective deployment motion look less like

$$
X_t \approx X_0 + V z_t
$$

with small residual, and more like a higher-rank process with substantial energy outside any small fixed subspace. In the notation of the paper, this is a regime where the residual term behind the low-rank specialization is not plausibly negligible.

Operationally, that means a small $V$ is likely to miss too much of the real deployment motion, and isotropic smoothing becomes more competitive because the benchmark is not dominated by one stable directional channel.

### 2.3 Metro Interstate Traffic Volume

This benchmark split the metrics:

- In the earlier exploratory run, `isotropic` had the best deployment MSE.
- DTR had the best volatility.

That is useful diagnostically, but it is not the kind of result that supports the paper's clean directional story.

Why it was not well suited:

Traffic volume is driven by several interacting factors at once:

- weather,
- time of day,
- day-of-week and holiday structure,
- commuting patterns,
- latent event regimes.

A benchmark like that need not violate the theorem, but it weakens the low-rank design implication. If deployment risk responds to many interacting directions, then a broad Jacobian penalty can be as effective as a targeted one, or better on average error even if DTR reduces the path variance.

The key lesson is:

> If DTR and isotropic smoothing win different metrics, the dataset is probably not a clean low-rank directional benchmark.

### 2.4 Individual Household Electric Power Consumption

This benchmark also came out mixed:

- In the earlier exploratory run, DTR had the best deployment MSE.
- `isotropic` had the best volatility.

Why it was not well suited:

The household series is affected by hidden behavioral state, not just exogenous measured covariates. A useful informal model is

$$
Y_t = g(X_t, H_t),
$$

where $X_t$ is the observed covariate vector and $H_t$ is a latent occupancy / appliance-usage regime. If $H_t$ contributes strongly and is only weakly observed through $X_t$, then controlling

$$
\|J_f(X)V\|_F^2
$$

need not control the important part of the deployment path. In that regime, broader smoothness can be just as valuable as directional smoothness, and sometimes more so.

So this was not a "DTR fails completely" dataset. It was a dataset where the observed covariates did not give a clean enough low-rank handle on the real temporal variation.

### 2.5 Bike Sharing

Bike Sharing was not a failure case, but it was still discarded from the manuscript.

Earlier exploratory summary:

- Deployment MSE: `DTR` `15550`, `isotropic` `15629`, `standard` `15984`.
- Volatility: `DTR` `1.96e7`, `isotropic` `2.01e7`, `standard` `2.05e7`.

So DTR did win, but only narrowly.

Why it was still dropped:

- The margin over isotropic smoothing was modest.
- The task is more aggregated demand forecasting than direct sensor or exogenous covariate drift.
- Once Tetouan was available, Bike Sharing no longer added much incremental evidence.

This dataset therefore belongs in a different category:

> DTR was plausible and mildly helpful, but the benchmark was not strong enough or distinctive enough to justify main-paper space.

## 3. Why the two retained benchmarks are well suited

### 3.1 UCI Air Quality

Air Quality is a good DTR benchmark because it is close to the motivating regime in the paper:

- The target is a fixed scalar regression target, $\mathrm{CO(GT)}$.
- The input covariates include sensor channels plus weather variables.
- The deployment story is naturally interpreted as sensor drift plus environmental covariate motion.
- The blockwise unlabeled covariates are meaningful for drift-subspace estimation.

That is much closer to the intended setting

$$
X_t \approx X_0 + V z_t + \varepsilon_t
$$

with a small number of dominant directions than the discarded mixed-shift benchmarks.

The final manuscript uses the target-orthogonal sensor subspace rather than the earlier all-covariate drift subspace. That distinction matters:

- With the target-orthogonal sensor subspace, validation selects $\lambda = 0.003$ for DTR.
- Mean deployment MSE improves from $0.449 \pm 0.069$ for standard training to $0.432 \pm 0.058$ for DTR.
- Mean volatility improves from $0.073 \pm 0.023$ to $0.069 \pm 0.020$.
- Terminal risk improves from $0.165 \pm 0.034$ to $0.161 \pm 0.028$.
- In paired seed comparisons against standard training, DTR improves deployment MSE in `9 / 10` seeds and volatility in `9 / 10` seeds.

The earlier all-covariate subspace remains useful as a cautionary ablation. It selected a larger penalty, $\lambda = 0.08$, reduced block-direction gain, but worsened deployment MSE and volatility. The lesson is the same as the synthetic misspecification study: DTR is useful when the estimated subspace captures nuisance motion rather than broad covariate motion or target signal.

### 3.2 UCI Tetouan City power consumption

Tetouan is the strongest real-data DTR benchmark we found.

Why it is well suited:

- The target is again a fixed scalar regression target: Zone 1 power consumption.
- The covariates are explicit exogenous drivers: temperature, humidity, wind speed, and diffuse-flow channels.
- The chronology is clean: train on January-April, validate on May-June, deploy on July-December.
- The dominant motion is plausibly seasonal and weather-driven, which makes a low-dimensional drift description much more reasonable than in the discarded multi-regime benchmarks.

In other words, Tetouan looks much closer to a setting where a small estimated $V$ genuinely captures the deployment path the model will traverse.

The validation-selected numbers are strong:

- Validation-selected DTR setting: $\lambda = 10^{-2}$.
- Deployment MSE: `standard` $(1.08 \pm 1.11) \times 10^8$, `isotropic` $(1.01 \pm 0.81) \times 10^8$, `DTR` $(6.82 \pm 4.87) \times 10^7$.
- Volatility: `standard` $(1.07 \pm 1.66) \times 10^{16}$, `isotropic` $(7.20 \pm 10.24) \times 10^{15}$, `DTR` $(3.07 \pm 4.32) \times 10^{15}$.
- DTR improves volatility in `8 / 10` paired seeds against both standard training and isotropic smoothing.

So Tetouan is not just directionally favorable in theory. It is a case where the directional bias was materially better than both plain ERM and isotropic Jacobian smoothing on the main deployment metrics.

## 4. Practical guidance: when not to use DTR

The benchmark search suggests a simple rule of thumb.

DTR is a poor default when one or more of the following are true:

- **Label mix is moving.** If unlabeled block means are strongly affected by $p_t(y)$, the estimated drift subspace is confounded.
- **The drift is high-rank or regime-switching.** If many directions carry material deployment motion, isotropic smoothing becomes more competitive.
- **Hidden state dominates observed covariates.** If temporal variation flows through latent regimes more than through measured exogenous drivers, directional covariate control is not enough.
- **Metrics split across methods.** If DTR wins volatility but isotropic wins deployment error, the benchmark is probably not aligned enough with the low-rank directional story to be a good showcase.

Conversely, DTR is a strong candidate when:

- the task is fixed over time,
- the covariates are explicit and physically interpretable,
- the deployment drift is plausibly concentrated in a few dominant directions,
- and a frozen-model deployment protocol is the actual use case.

That is the main reason Air Quality and Tetouan were retained, while the other exploratory branches were not.
