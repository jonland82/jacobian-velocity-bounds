from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math

import numpy as np
import pandas as pd
import sympy as sp


@dataclass
class CheckResult:
    slug: str
    title: str
    category: str
    method: str
    passed: bool
    summary: str
    details: list[str] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)
    math_blocks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _to_builtin(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def check_poincare_sharpness() -> CheckResult:
    t, horizon = sp.symbols("t horizon", positive=True)
    r = sp.cos(sp.pi * t / horizon)
    mean = sp.simplify(sp.integrate(r, (t, 0, horizon)) / horizon)
    variance = sp.simplify(sp.integrate((r - mean) ** 2, (t, 0, horizon)) / horizon)
    rhs = sp.simplify(
        horizon / sp.pi**2 * sp.integrate(sp.diff(r, t) ** 2, (t, 0, horizon))
    )
    gap = sp.simplify(rhs - variance)
    passed = mean == 0 and gap == 0
    return CheckResult(
        slug="poincare-sharpness",
        title="Poincare / Wirtinger sharpness",
        category="exact symbolic",
        method="sympy",
        passed=passed,
        summary="The first cosine eigenfunction attains equality in the time-domain bound.",
        details=[
            "Used r(t) = cos(pi t / T), the first zero-mean eigenfunction on [0, T].",
            "Verified the temporal mean is exactly zero.",
            "Verified Var_U(r(U)) and (T / pi^2) int_0^T (r'(t))^2 dt simplify to the same closed form.",
        ],
        metrics={
            "mean": sp.sstr(mean),
            "variance": sp.sstr(variance),
            "rhs": sp.sstr(rhs),
            "gap": sp.sstr(gap),
        },
        math_blocks=[
            r"\bar r = \frac{1}{T}\int_0^T \cos\!\left(\frac{\pi t}{T}\right) dt = 0",
            r"\mathrm{Var}_U(r(U)) = \frac{1}{T}\int_0^T \cos^2\!\left(\frac{\pi t}{T}\right) dt = \frac{1}{2}",
            r"\frac{T}{\pi^2}\int_0^T (r'(t))^2 dt = \frac{1}{2}",
        ],
    )


def check_jacobian_velocity_equality() -> CheckResult:
    x, t, horizon = sp.symbols("x t horizon", positive=True)
    f = sp.cos(sp.pi * x / horizon)
    r = f.subs(x, t)
    variance = sp.simplify(sp.integrate(r**2, (t, 0, horizon)) / horizon)
    jv_rhs = sp.simplify(
        horizon
        / sp.pi**2
        * sp.integrate((sp.diff(f, x).subs(x, t)) ** 2, (t, 0, horizon))
    )
    gap = sp.simplify(jv_rhs - variance)
    passed = gap == 0
    return CheckResult(
        slug="jv-equality",
        title="Jacobian-velocity theorem on a deterministic path",
        category="exact symbolic",
        method="sympy",
        passed=passed,
        summary="For X_t = t, g = f, and f(x) = cos(pi x / T), the theorem is exact with beta = 1.",
        details=[
            "The deployment path is deterministic, so the expectation operators collapse exactly.",
            "Because g = f, Assumption A3 holds with beta = 1 and |grad g dot Xdot| = |J_f Xdot|.",
            "The Jacobian-velocity right-hand side equals the exact volatility from the lemma-sharp example.",
        ],
        metrics={
            "variance": sp.sstr(variance),
            "jv_rhs": sp.sstr(jv_rhs),
            "gap": sp.sstr(gap),
        },
        math_blocks=[
            r"X_t = t,\qquad \dot X_t = 1,\qquad f(x) = g(x) = \cos\!\left(\frac{\pi x}{T}\right)",
            r"J_f(X_t)\dot X_t = -\frac{\pi}{T}\sin\!\left(\frac{\pi t}{T}\right)",
            r"\mathrm{Var}_U(r(U)) = \frac{T}{\pi^2}\int_0^T \left\|J_f(X_t)\dot X_t\right\|^2 dt = \frac{1}{2}",
        ],
    )


def check_composition_chain_rule() -> CheckResult:
    x1, x2, v1, v2, z = sp.symbols("x1 x2 v1 v2 z", real=True)
    direction = sp.Matrix([v1, v2])
    f = x1**2 + x1 * x2 + sp.sin(x2)
    g = sp.atan(f)
    grad_g = sp.Matrix([sp.diff(g, x1), sp.diff(g, x2)])
    grad_f = sp.Matrix([sp.diff(f, x1), sp.diff(f, x2)])
    lhs = sp.simplify(grad_g.dot(direction))
    rhs = sp.simplify(sp.diff(sp.atan(z), z).subs(z, f) * grad_f.dot(direction))
    difference = sp.simplify(lhs - rhs)
    passed = difference == 0
    return CheckResult(
        slug="composition-chain-rule",
        title="Composition case in Remark A3",
        category="exact symbolic",
        method="sympy",
        passed=passed,
        summary="Directional derivatives of g = h o f factor exactly into h'(f) and the score Jacobian term.",
        details=[
            "Used a nontrivial smooth score field f(x1, x2) = x1^2 + x1 x2 + sin(x2).",
            "Used h(z) = atan(z), so g(x1, x2) = atan(f(x1, x2)).",
            "Simplified grad g dot v - h'(f) grad f dot v to zero exactly.",
        ],
        metrics={"difference": sp.sstr(difference)},
        math_blocks=[
            r"g(x) = h(f(x)),\qquad h(z) = \arctan(z)",
            rf"\nabla g(x)\cdot v = h'(f(x))\, \nabla f(x)\cdot v",
        ],
    )


def check_hazard_rank1_bookkeeping() -> CheckResult:
    a, r, c, s = sp.symbols("a r c s", real=True)
    m11, m12, m21, m22 = sp.symbols("m11 m12 m21 m22", real=True)

    matrix = sp.Matrix([[m11, m12], [m21, m22]])
    v = sp.Matrix([1, 0])
    u = sp.Matrix([0, 1])
    delta = sp.Matrix([a, r])
    v_t = c * v + s * u

    s_sq_gap = sp.simplify(delta.dot(delta) - (a**2 + r**2))

    aligned = sp.simplify((matrix * v).dot(matrix * v))
    orthogonal = sp.simplify((matrix * u).dot(matrix * u))
    overlap = sp.simplify((matrix * v).dot(matrix * u))
    g_gap = sp.simplify(
        sp.expand((matrix * v_t).dot(matrix * v_t))
        - sp.expand(c**2 * aligned + s**2 * orthogonal + 2 * c * s * overlap)
    )
    h_gap = sp.simplify(
        sp.expand(delta.dot(delta) * (matrix * v_t).dot(matrix * v_t))
        - sp.expand((a**2 + r**2) * (c**2 * aligned + s**2 * orthogonal + 2 * c * s * overlap))
    )

    passed = s_sq_gap == 0 and g_gap == 0 and h_gap == 0
    return CheckResult(
        slug="hazard-rank1-bookkeeping",
        title="Rank-1 hazard-score bookkeeping",
        category="exact symbolic",
        method="sympy",
        passed=passed,
        summary="The pointwise algebra behind Proposition 1 is exact; the expectation form follows by averaging these identities.",
        details=[
            "Worked in an orthonormal basis with v = e1, u = e2, Delta mu / Delta = (a, r), and a generic 2 x 2 Jacobian matrix.",
            "Verified s_t^2 = |a|^2 + |r|^2 from the orthogonal block-drift decomposition.",
            "Verified the directional energy splits into aligned, orthogonal, and overlap terms with coefficients c^2, s^2, and 2cs.",
        ],
        metrics={
            "s_sq_gap": sp.sstr(s_sq_gap),
            "g_gap": sp.sstr(g_gap),
            "h_gap": sp.sstr(h_gap),
        },
        math_blocks=[
            r"\frac{\Delta \mu_t}{\Delta} = a v + r u,\qquad v_t = c v + s u,\qquad v^\top u = 0",
            r"s_t^2 = \left\|\frac{\Delta \mu_t}{\Delta}\right\|^2 = |a|^2 + |r|^2",
            r"\|J_f v_t\|^2 = c^2\|J_f v\|^2 + s^2\|J_f u\|^2 + 2cs\langle J_f v, J_f u\rangle",
        ],
    )


def check_cross_entropy_derivative_bound() -> CheckResult:
    z, q = sp.symbols("z q", real=True)
    sigmoid = 1 / (1 + sp.exp(-z))
    h = -(q * sp.log(sigmoid) + (1 - q) * sp.log(1 - sigmoid))
    derivative = sp.simplify(sp.diff(h, z))
    identity_gap = sp.simplify(derivative - (sigmoid - q))

    z_grid = np.linspace(-12.0, 12.0, 241)
    q_grid = np.linspace(0.0, 1.0, 121)
    max_abs = 0.0
    for q_value in q_grid:
        sigmoid_values = 1.0 / (1.0 + np.exp(-z_grid))
        max_abs = max(max_abs, float(np.max(np.abs(sigmoid_values - q_value))))

    passed = identity_gap == 0 and max_abs <= 1.0 + 1e-12
    return CheckResult(
        slug="cross-entropy-derivative",
        title="Bernoulli cross-entropy derivative bound",
        category="exact symbolic",
        method="sympy + dense numeric sweep",
        passed=passed,
        summary="The soft-target Bernoulli cross-entropy derivative is sigma(z) - q, so its magnitude never exceeds 1.",
        details=[
            "Differentiated the scalar loss h(z) = -q log sigma(z) - (1-q) log(1-sigma(z)) exactly.",
            "Verified the closed form h'(z) = sigma(z) - q symbolically.",
            "Checked the beta = 1 bound over a dense grid in z and q.",
        ],
        metrics={
            "identity_gap": sp.sstr(identity_gap),
            "max_abs_derivative_on_grid": _to_builtin(max_abs),
        },
        math_blocks=[
            r"h(z) = -q \log \sigma(z) - (1-q)\log(1-\sigma(z))",
            r"h'(z) = \sigma(z) - q,\qquad |h'(z)| \le 1 \text{ for } q \in [0,1]",
        ],
    )


def check_corollary_randomized(trials: int = 5000, seed: int = 7) -> CheckResult:
    rng = np.random.default_rng(seed)
    d = 7
    k = 3
    out_dim = 5

    max_split_violation = -math.inf
    max_frobenius_violation = -math.inf
    max_combined_violation = -math.inf
    max_orthogonality_error = 0.0
    min_combined_slack = math.inf

    for _ in range(trials):
        raw_v = rng.normal(size=(d, k))
        v_basis, _ = np.linalg.qr(raw_v, mode="reduced")
        matrix = rng.normal(size=(out_dim, d))
        coeff = rng.normal(size=k)
        rho_raw = rng.normal(size=d)
        rho = rho_raw - v_basis @ (v_basis.T @ rho_raw)

        u = matrix @ (v_basis @ coeff)
        w = matrix @ rho
        lhs = float(np.linalg.norm(u + w) ** 2)
        split_rhs = float(2.0 * np.linalg.norm(u) ** 2 + 2.0 * np.linalg.norm(w) ** 2)
        frob_rhs = float(
            2.0 * (np.linalg.norm(matrix @ v_basis, ord="fro") ** 2) * np.linalg.norm(coeff) ** 2
            + 2.0 * np.linalg.norm(w) ** 2
        )

        max_split_violation = max(max_split_violation, lhs - split_rhs)
        max_frobenius_violation = max(
            max_frobenius_violation,
            float(np.linalg.norm(u) ** 2)
            - float(np.linalg.norm(matrix @ v_basis, ord="fro") ** 2 * np.linalg.norm(coeff) ** 2),
        )
        max_combined_violation = max(max_combined_violation, lhs - frob_rhs)
        max_orthogonality_error = max(max_orthogonality_error, float(np.linalg.norm(v_basis.T @ rho)))
        min_combined_slack = min(min_combined_slack, frob_rhs - lhs)

    tolerance = 1e-10
    passed = (
        max_split_violation <= tolerance
        and max_frobenius_violation <= tolerance
        and max_combined_violation <= tolerance
        and max_orthogonality_error <= 1e-10
    )
    return CheckResult(
        slug="corollary-randomized",
        title="Low-rank corollary inequalities",
        category="numerical",
        method="numpy randomized stress test",
        passed=passed,
        summary="The triangle and Frobenius inequalities used in the low-rank corollary hold across randomized orthonormal decompositions.",
        details=[
            "Generated random orthonormal drift bases V with QR factorization.",
            "Projected residual drift rho into the orthogonal complement of span(V).",
            "Checked the split inequality, the Frobenius inequality, and the combined corollary bound trial by trial.",
        ],
        metrics={
            "trials": trials,
            "max_split_violation": _to_builtin(max_split_violation),
            "max_frobenius_violation": _to_builtin(max_frobenius_violation),
            "max_combined_violation": _to_builtin(max_combined_violation),
            "min_combined_slack": _to_builtin(min_combined_slack),
            "max_orthogonality_error": _to_builtin(max_orthogonality_error),
        },
        math_blocks=[
            r"\|u+w\|^2 \le 2\|u\|^2 + 2\|w\|^2",
            r"\|MVa\|^2 \le \|MV\|_F^2 \|a\|^2",
            r"\begin{aligned} \|M(Va+\rho)\|^2 &\le 2\|MV\|_F^2\|a\|^2 \\ &\quad + 2\|M\rho\|^2 \end{aligned}",
        ],
    )


def check_theorem_numeric_expectation() -> CheckResult:
    horizon = 1.0
    times = np.linspace(0.0, horizon, 4001)
    shifts = np.array([-0.2, 0.0, 0.15], dtype=float)
    probs = np.array([0.2, 0.5, 0.3], dtype=float)

    def f(x: np.ndarray) -> np.ndarray:
        return np.cos(math.pi * x)

    def jacobian(x: np.ndarray) -> np.ndarray:
        return -math.pi * np.sin(math.pi * x)

    def h(z: np.ndarray) -> np.ndarray:
        return np.arctan(z)

    def hprime(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + z**2)

    x_grid = times[:, None] + shifts[None, :]
    f_values = f(x_grid)
    g_values = h(f_values)
    dg_dt = hprime(f_values) * jacobian(x_grid)
    jv_sq = jacobian(x_grid) ** 2

    risk = g_values @ probs
    risk_prime = dg_dt @ probs
    expected_jv_sq = jv_sq @ probs

    risk_mean = float(np.trapezoid(risk, times) / horizon)
    volatility = float(np.trapezoid((risk - risk_mean) ** 2, times) / horizon)
    chain_bound = float(horizon / math.pi**2 * np.trapezoid(risk_prime**2, times))
    jv_bound = float(horizon / math.pi**2 * np.trapezoid(expected_jv_sq, times))
    beta_upper = float(np.max(hprime(np.linspace(-1.0, 1.0, 2001))))

    tolerance = 5e-8
    passed = volatility <= chain_bound + tolerance and chain_bound <= jv_bound + tolerance
    return CheckResult(
        slug="theorem-numeric-expectation",
        title="Jacobian-velocity theorem with a nontrivial expectation",
        category="numerical",
        method="dense quadrature on a smooth finite-mixture path",
        passed=passed,
        summary="A smooth mixture path satisfies the full inequality chain Var <= derivative-energy bound <= Jacobian-velocity bound.",
        details=[
            "Used X_t = t + Z with a three-point discrete random offset Z and deterministic velocity Xdot = 1.",
            "Used f(x) = cos(pi x) and g(x) = atan(f(x)), so A3 holds with beta <= 1.",
            "Integrated the continuous-time quantities on a dense grid to verify the theorem in a genuine expectation setting.",
        ],
        metrics={
            "beta_upper": _to_builtin(beta_upper),
            "volatility": _to_builtin(volatility),
            "chain_bound": _to_builtin(chain_bound),
            "jv_bound": _to_builtin(jv_bound),
            "chain_minus_volatility": _to_builtin(chain_bound - volatility),
            "jv_minus_chain": _to_builtin(jv_bound - chain_bound),
        },
        math_blocks=[
            r"X_t = t + Z,\qquad Z \in \{-0.2, 0, 0.15\}",
            r"f(x) = \cos(\pi x),\qquad g(x) = \arctan(f(x)),\qquad \beta \le 1",
            r"\begin{aligned} \mathrm{Var}_U(r(U)) &\le \frac{T}{\pi^2}\int_0^T (r'(t))^2 dt \\ &\le \frac{T}{\pi^2}\int_0^T \mathbb{E}\!\left[\|J_f(X_t)\dot X_t\|^2\right] dt \end{aligned}",
        ],
    )


def _artifact_check(
    repo_root: Path,
    relative_path: str,
    title: str,
    summary: str,
) -> CheckResult:
    csv_path = repo_root / relative_path
    frame = pd.read_csv(csv_path)
    if "volatility" not in frame.columns:
        raise ValueError(f"Expected a volatility column in {csv_path}")

    bound_columns = [col for col in ("bound_fd", "bound_chain", "bound_jv") if col in frame.columns]
    slacks = {col: float((frame[col] - frame["volatility"]).min()) for col in bound_columns}
    tolerance = 5e-5
    passed = all(slack >= -tolerance for slack in slacks.values())

    details = [
        f"Loaded {len(frame)} rows from {relative_path}.",
        "Checked volatility <= bound_fd, volatility <= bound_chain, and volatility <= bound_jv wherever those columns exist.",
        "Allowed an absolute tolerance of 5e-5 for cached numerical summaries, since those bounds come from finite grids and Monte Carlo estimates rather than exact algebra.",
    ]
    metrics: dict[str, object] = {"rows": int(len(frame)), "absolute_tolerance": tolerance}
    for key, value in slacks.items():
        metrics[f"min_slack_{key}"] = _to_builtin(value)

    return CheckResult(
        slug=csv_path.stem,
        title=title,
        category="artifact consistency",
        method="pandas cached-summary validation",
        passed=passed,
        summary=summary,
        details=details,
        metrics=metrics,
        math_blocks=[
            r"\begin{aligned} \mathrm{volatility} &\le \mathrm{bound\_fd} \\ \mathrm{volatility} &\le \mathrm{bound\_chain} \\ \mathrm{volatility} &\le \mathrm{bound\_jv} \end{aligned}",
        ],
    )


def run_all_checks(repo_root: Path) -> list[CheckResult]:
    return [
        check_poincare_sharpness(),
        check_jacobian_velocity_equality(),
        check_composition_chain_rule(),
        check_hazard_rank1_bookkeeping(),
        check_cross_entropy_derivative_bound(),
        check_corollary_randomized(),
        check_theorem_numeric_expectation(),
        _artifact_check(
            repo_root,
            "figures/synthetic_theorem_summary.csv",
            "Synthetic theorem summary bounds",
            "Every cached synthetic theorem run satisfies the finite-difference, chain-rule, and Jacobian-velocity bounds.",
        ),
        _artifact_check(
            repo_root,
            "figures/synthetic_directional_comparison_summary.csv",
            "Directional comparison bounds",
            "Every cached directional-comparison run preserves the volatility upper bounds used in the paper.",
        ),
        _artifact_check(
            repo_root,
            "figures/synthetic_directional_misspecification_summary.csv",
            "Misspecification bounds",
            "Every cached misspecification run preserves the volatility upper bounds under subspace rotation.",
        ),
    ]
