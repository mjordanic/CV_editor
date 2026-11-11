"""Statistics for paired arm comparisons, noise components and sample sizes. numpy only; scipy is used
for the Wilcoxon signed-rank test, rank correlations and normal/t quantiles when it is installed
(the `analysis` dependency group), with exact or normal-approximation fallbacks otherwise.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

try:  # optional: the `analysis` dependency group
    from scipy import stats as _sps
except ImportError:  # pragma: no cover - exercised only without scipy
    _sps = None

_Z = NormalDist()


def bootstrap_mean_ci(values, clusters=None, n_boot: int = 10000, level: float = 0.95, seed: int = 0):
    """(mean, low, high): percentile bootstrap CI of the mean. With `clusters` (one label per value)
    whole clusters are resampled, e.g. all cells of one persona together."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, x.size, size=(n_boot, x.size))
        boots = x[idx].mean(axis=1)
    else:
        labels = np.asarray(clusters)
        groups = [x[labels == g] for g in dict.fromkeys(labels.tolist())]
        sums = np.array([g.sum() for g in groups])
        sizes = np.array([g.size for g in groups])
        pick = rng.integers(0, len(groups), size=(n_boot, len(groups)))
        boots = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    a = (1 - level) / 2
    return (float(x.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a)))


def sign_test(diffs) -> dict:
    """Exact two-sided sign test; ties (zero differences) are dropped. Wins/ties/losses are for B (diff > 0)."""
    d = np.asarray(diffs, dtype=float)
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    ties = int(d.size - wins - losses)
    n = wins + losses
    if n == 0:
        return {"wins": wins, "ties": ties, "losses": losses, "p": 1.0}
    k = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"wins": wins, "ties": ties, "losses": losses, "p": p}


def signflip_test(diffs, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided paired permutation (sign-flip) test of mean difference = 0. Exact up to 16 pairs."""
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    obs = abs(d.mean())
    if d.size <= 16:
        signs = np.array(np.meshgrid(*[[-1, 1]] * d.size)).reshape(d.size, -1).T
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, d.size))
    means = np.abs((signs * d).mean(axis=1))
    return float((means >= obs - 1e-12).mean())


def persona_signflip_test(diffs, clusters) -> float:
    """Sign-flip test that flips every cluster's differences together (e.g. the three input sets of one persona):
    the clusters, not the cells, are the independent units. Exact up to 16 clusters (13 personas: 8,192)."""
    d, labels = np.asarray(diffs, dtype=float), np.asarray(clusters)
    return signflip_test([d[labels == g].sum() for g in dict.fromkeys(labels.tolist())])


def wilcoxon(diffs) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value (Pratt's handling of zero differences), or None without scipy
    or without a non-zero difference."""
    d = np.asarray(diffs, dtype=float)
    if _sps is None or not np.any(d != 0):
        return None
    return float(_sps.wilcoxon(d, zero_method="pratt").pvalue)


def holm(pvalues: dict) -> dict:
    """Holm-adjusted p-values for a family of tests ({name: p}); None values are skipped."""
    items = sorted(((p, k) for k, p in pvalues.items() if p is not None))
    m = len(items)
    out, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return {k: out.get(k) for k in pvalues}


def paired_summary(diffs, clusters=None, seed: int = 0) -> dict:
    """Everything the notebook reports for one paired comparison B - A."""
    d = np.asarray(diffs, dtype=float)
    n = int(d.size)
    mean, lo, hi = bootstrap_mean_ci(d, seed=seed)
    out = {"n_pairs": n, "mean_diff": mean, "ci_low": lo, "ci_high": hi,
           "median_diff": float(np.median(d)) if n else math.nan,
           "sd_diff": float(d.std(ddof=1)) if n >= 2 else math.nan}
    out["dz"] = out["mean_diff"] / out["sd_diff"] if n >= 2 and out["sd_diff"] > 0 else math.nan
    if clusters is not None and n:
        _, clo, chi = bootstrap_mean_ci(d, clusters=clusters, seed=seed)
        out.update(ci_low_cluster=clo, ci_high_cluster=chi, p_signflip_cluster=persona_signflip_test(d, clusters))
    st = sign_test(d)
    out.update(wins_b=st["wins"], ties=st["ties"], wins_a=st["losses"], p_sign=st["p"],
               p_wilcoxon=wilcoxon(d), p_signflip=signflip_test(d, seed=seed) if n else math.nan)
    return out


def wilson_ci(successes: int, n: int, level: float = 0.95):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (math.nan, math.nan, math.nan)
    z = _Z.inv_cdf(1 - (1 - level) / 2)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (p, min(p, max(0.0, centre - half)), max(p, min(1.0, centre + half)))


# --- noise components --------------------------------------------------------------------------------
def pooled_within_var(groups) -> tuple[float, int]:
    """(pooled within-group variance, degrees of freedom) over groups of repeated values."""
    ss, df = 0.0, 0
    for g in groups:
        g = np.asarray(g, dtype=float)
        if g.size >= 2:
            ss += float(((g - g.mean()) ** 2).sum())
            df += g.size - 1
    return (ss / df if df else math.nan, df)


def writer_var(groups, ks, var_grader: float) -> float:
    """Writer variance from repeat groups (run values `groups`, each run the mean of `ks` gradings): the pooled
    within-group variance is writer variance plus grader variance x the mean of 1/k, so that part is taken off
    (with var_grader from re-gradings; never below 0). NaN without repeats or without a grader variance."""
    var_w, df = pooled_within_var(groups)
    if not df or math.isnan(var_grader):
        return math.nan
    weights = [(len(g) - 1, float(np.mean([1 / k for k in kk]))) for g, kk in zip(groups, ks) if len(g) >= 2]
    factor = sum(w * f for w, f in weights) / sum(w for w, _ in weights)
    return max(0.0, var_w - factor * var_grader)


def var_ci(var: float, df: int, level: float = 0.95):
    """Chi-square CI for a variance with df degrees of freedom (needs scipy; else (nan, nan))."""
    if _sps is None or not df or math.isnan(var):
        return (math.nan, math.nan)
    a = (1 - level) / 2
    return (df * var / _sps.chi2.ppf(1 - a, df), df * var / _sps.chi2.ppf(a, df))


def diff_sd(sd_interaction: float, sd_writer: float, sd_grader: float, reps: int = 1, gradings: int = 1) -> float:
    """SD of one cell's paired difference B - A when each arm's cell value is the mean of `reps` sessions,
    each the mean of `gradings` gradings: interaction (arm x cell) + writer noise + grader noise."""
    return math.sqrt(sd_interaction ** 2 + 2 * sd_writer ** 2 / reps + 2 * sd_grader ** 2 / (reps * gradings))


def pairs_needed(delta: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Pairs for a two-sided paired t-test to detect a mean difference `delta` (t-corrected iteration)."""
    if delta <= 0 or not sd or math.isnan(sd):
        return 0
    za, zb = _Z.inv_cdf(1 - alpha / 2), _Z.inv_cdf(power)
    n = max(3, math.ceil(((za + zb) * sd / delta) ** 2))
    if _sps is None:
        return n
    for _ in range(50):
        ta, tb = _sps.t.ppf(1 - alpha / 2, n - 1), _sps.t.ppf(power, n - 1)
        new = max(3, math.ceil(((ta + tb) * sd / delta) ** 2))
        if new == n:
            break
        n = new
    return n


def power_paired(n: int, delta: float, sd: float, alpha: float = 0.05) -> float:
    """Approximate power of a two-sided paired t-test with n pairs (normal approximation)."""
    if n < 2 or not sd:
        return math.nan
    za = _Z.inv_cdf(1 - alpha / 2)
    shift = delta * math.sqrt(n) / sd
    return _Z.cdf(shift - za) + _Z.cdf(-shift - za)


def simulate_diffs(n_pairs: int, delta: float, sd_interaction: float, sd_writer: float, sd_grader: float,
                   reps: int = 1, gradings: int = 1, base: float = 7.4, rng=None, n_sim: int = 1) -> np.ndarray:
    """Simulated paired cell differences B - A, shape (n_sim, n_pairs), with integer 0-10 grades as the grader
    gives them: a latent cell quality, an arm x cell interaction, writer noise per session and grader noise per
    grading; every grading is rounded and clipped to 0-10, then averaged over gradings and repeats."""
    rng = rng if rng is not None else np.random.default_rng(0)
    cell = base + rng.normal(0, 0.8, (n_sim, 1, n_pairs, 1))
    inter = rng.normal(0, sd_interaction / math.sqrt(2), (2, n_sim, 1, n_pairs, 1))

    def arm(shift, k):
        sess = cell + shift + inter[k] + rng.normal(0, sd_writer, (n_sim, reps, n_pairs, 1))
        grades = np.clip(np.rint(sess + rng.normal(0, sd_grader, (n_sim, reps, n_pairs, gradings))), 0, 10)
        return grades.mean(axis=(1, 3))

    return arm(delta, 1) - arm(0.0, 0)


def calibrate_interaction(target_sd: float, sd_writer: float, sd_grader: float, n: int = 20000, seed: int = 0) -> float:
    """The latent interaction SD at which simulated integer-grade differences (r = 1, k = 1) have the observed
    SD. Rounding to integers adds spread of its own, so the observed SD must not be used as the latent one."""
    grid = np.linspace(0, target_sd, 41)
    sds = [simulate_diffs(n, 0.0, s, sd_writer, sd_grader, rng=np.random.default_rng(seed)).std() for s in grid]
    return float(grid[int(np.argmin(np.abs(np.array(sds) - target_sd)))])


def simulate_power(n_pairs: int, delta: float, sd_interaction: float, sd_writer: float, sd_grader: float,
                   reps: int = 1, gradings: int = 1, alpha: float = 0.05, base: float = 7.4,
                   n_sim: int = 2000, seed: int = 0) -> dict:
    """Monte Carlo power of the Wilcoxon test (if scipy) and the sign test with integer grades (simulate_diffs)."""
    d = simulate_diffs(n_pairs, delta, sd_interaction, sd_writer, sd_grader, reps, gradings, base,
                       np.random.default_rng(seed), n_sim)
    wins, losses = (d > 0).sum(axis=1), (d < 0).sum(axis=1)
    if _sps is None:
        p_sign = np.array([sign_test(row)["p"] for row in d])
        return {"wilcoxon": math.nan, "sign": float((p_sign < alpha).mean())}
    n = wins + losses
    p_sign = np.where(n > 0, np.minimum(1.0, 2 * _sps.binom.cdf(np.minimum(wins, losses), np.maximum(n, 1), 0.5)), 1.0)
    nonzero = n > 0
    p_w = np.ones(n_sim)
    if nonzero.any():
        with np.errstate(all="ignore"):
            p_w[nonzero] = _sps.wilcoxon(d[nonzero], zero_method="pratt", axis=1, method="approx").pvalue
    p_w = np.nan_to_num(p_w, nan=1.0)
    return {"wilcoxon": float((p_w < alpha).mean()), "sign": float((p_sign < alpha).mean())}


# --- agreement between graders -----------------------------------------------------------------------
def weighted_kappa(a, b, labels=range(11)) -> float:
    """Quadratic-weighted Cohen's kappa for integer grades (values are rounded)."""
    labels = list(labels)
    idx = {v: i for i, v in enumerate(labels)}
    k = len(labels)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[int(np.rint(x))], idx[int(np.rint(y))]] += 1
    if obs.sum() == 0:
        return math.nan
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    den = (w * exp).sum()
    return float(1 - (w * obs).sum() / den) if den else math.nan


def agreement(a, b) -> dict:
    """Agreement of two graders on the same runs: exact and within-one agreement, mean bias (B - A) with a
    bootstrap CI, Spearman and Kendall rank correlation, quadratic-weighted kappa."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = {"n": int(a.size)}
    if a.size == 0:
        return out
    bias, lo, hi = bootstrap_mean_ci(b - a)
    out.update(exact=float((np.rint(a) == np.rint(b)).mean()), within_1=float((np.abs(a - b) <= 1).mean()),
               bias=bias, bias_low=lo, bias_high=hi, kappa_qw=weighted_kappa(a, b))
    if _sps is not None and a.size >= 3 and a.std() > 0 and b.std() > 0:
        out.update(spearman=float(_sps.spearmanr(a, b).statistic), kendall=float(_sps.kendalltau(a, b).statistic))
    return out
