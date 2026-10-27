"""Selection-aware correction and certified savings floors.

Problem. A recommender picks w = argmin_i mu_hat_i over n functionally
equivalent candidates and reports  claimed_saving = baseline - mu_hat_w.
Because w was chosen for having a low ESTIMATE, mu_hat_w is optimistically
biased: E[mu_w | w selected] > mu_hat_w. So the claimed saving exceeds the
realized saving systematically (the optimizer's curse).

Estimators implemented
----------------------
naive               report mu_hat_w as-is (what deployed tools do)
ec3_band            mu_hat_w scaled by EC3's deployed fixed uncertainty_factor
                    (~1.095, invariant to class/geography) -- deployed baseline
eb_shrink           empirical-Bayes / Tweedie shrinkage of mu_hat toward the
                    class mean BEFORE selection, using sigma^2 (predictive
                    noise) and tau^2 (between-product spread). Shrinking the
                    estimates removes most of the selection-induced optimism
                    because extreme estimates are shrunk most.
split_select        select on one half of the evidence, estimate on the other.
                    Unbiased by construction; the honest reference point.
sel_conformal       distribution-free CERTIFIED FLOOR. Calibrate the residual
                    quantile ON THE SELECTION EVENT: run the whole selection
                    procedure over calibration candidate sets, collect the
                    realized-minus-claimed residuals OF THE WINNERS ONLY, and
                    take the (1-alpha) quantile. Valid under exchangeability of
                    candidate sets, and targets exactly the biased distribution
                    rather than the marginal one.

All estimators operate on log GWP; savings are reported on the natural scale.

Supplies: `CalibrationMap` is the alpha/beta fit behind RQ2 and Proposition 1, and it
is the map every reporting rule in Appendix D is built on. `adaptive_correction` and
`SelectionConformal` support the supplementary corrected reruns in
``experiments/supplementary``. The remaining estimators are the deployed and
textbook baselines the manuscript positions against; they are kept together so the
comparison set stays visible in one place.
"""
import numpy as np


# --------------------------------------------------------------------------
# point estimators (operate on the predicted log-GWP of a candidate set)
# --------------------------------------------------------------------------
def eb_shrink(pred_log, sigma2, prior_mean=None, tau2=None):
    """Empirical-Bayes posterior mean under a normal-normal model.

    pred_log : predicted log-GWP for the candidate set
    sigma2   : predictive error variance (per-candidate or scalar)
    tau2     : between-product variance; estimated from the spread of pred_log
               net of predictive noise when not supplied.
    """
    pred_log = np.asarray(pred_log, dtype=float)
    m = float(np.mean(pred_log)) if prior_mean is None else float(prior_mean)
    if tau2 is None:
        tau2 = max(float(np.var(pred_log)) - float(np.mean(sigma2)), 1e-8)
    shrink = tau2 / (tau2 + sigma2)          # -> 0 when noise dominates
    return m + shrink * (pred_log - m)


def eb_shrink_pop(pred_log, sigma2, prior_mean, tau2):
    """Empirical Bayes with POPULATION prior parameters.

    eb_shrink() estimates tau^2 from the candidate set itself, which is noisy
    for small n and drives over-shrinkage at large n. tau^2 and the prior mean
    are properties of the material class, not of one query, so estimate them
    once on the calibration population and pass them in.
    """
    pred_log = np.asarray(pred_log, dtype=float)
    shrink = tau2 / (tau2 + sigma2)
    return prior_mean + shrink * (pred_log - prior_mean)


class CalibrationMap:
    """Stage 1 — recalibrate out-of-domain predictions.

    A model fitted on one set of manufacturers produces OVER-DISPERSED
    predictions on unseen manufacturers: the regression of truth on prediction
    has slope beta < 1. Empirically beta falls from ~0.89 with full evidence to
    ~0.26 with 1% of it, so the over-dispersion is worst exactly where evidence
    is thin. argmin selection then preferentially picks products whose
    predictions are spuriously low.

    Fit on a calibration block of manufacturers disjoint from both training and
    evaluation (Van Houwelingen / Cox calibration regression).

    The calibration corpus is manufacturer-skewed (a handful of producers
    publish tens of thousands of declarations each). Plain row-level OLS lets
    whichever manufacturer happens to be biggest in a given calibration fold
    dominate the slope estimate -- measured to swing beta from 0.03 to 0.98
    across folds that mostly differ only in which large manufacturer they
    contain. Pass `groups` (e.g. manufacturer id per row) to fit alpha/beta by
    weighted least squares with each group weighted equally regardless of its
    row count, exactly the same discipline `splits.balanced_group_blocks`
    already applies to fold construction. sigma2/tau2 remain row-level
    (unweighted): they characterise the residual a randomly drawn candidate
    faces under selection, which is a row-level quantity by construction of
    the selection procedure, not a supplier-transfer quantity.
    """

    def __init__(self):
        self.alpha = 0.0
        self.beta = 1.0
        self.sigma2 = None   # residual variance AFTER recalibration
        self.tau2 = None     # variance of recalibrated predictions

    def fit(self, pred, truth, groups=None):
        p = np.asarray(pred, dtype=float)
        t = np.asarray(truth, dtype=float)
        A = np.vstack([np.ones_like(p), p]).T
        if groups is None:
            coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        else:
            g = np.asarray(groups)
            _, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
            w = 1.0 / counts[inv]                      # each group sums to weight 1
            w = w / w.mean()                            # keep lstsq well-scaled
            sw = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(A * sw[:, None], t * sw, rcond=None)
        self.alpha, self.beta = float(coef[0]), float(coef[1])
        adj = self.transform(p)
        self.sigma2 = float(np.var(t - adj))
        self.tau2 = max(float(np.var(adj)), 1e-8)
        return self

    def transform(self, pred):
        return self.alpha + self.beta * np.asarray(pred, dtype=float)


_AN_CACHE = {}


def expected_min_normal(n, reps=400_000, seed=0):
    """a_n = -E[min of n i.i.d. standard normals]  (a_n > 0 for n >= 2).

    The textbook sqrt(2 ln n) asymptotic is poor in the range that matters
    here: it gives 1.177 at n=2 where the exact value is 1/sqrt(pi) = 0.564.
    Small n is the common case for material substitution, so use exact values
    where they are known and Monte Carlo elsewhere (cached, deterministic).
    """
    n = int(n)
    if n < 2:
        return 0.0
    if n == 2:
        return 1.0 / np.sqrt(np.pi)            # exact
    if n == 3:
        return 3.0 / (2.0 * np.sqrt(np.pi))    # exact
    if n in _AN_CACHE:
        return _AN_CACHE[n]
    rng = np.random.default_rng(seed)
    a = float(-np.mean(rng.standard_normal((reps, n)).min(axis=1)))
    _AN_CACHE[n] = a
    return a


def adaptive_correction(sigma2, n, tau2, kappa=1.0):
    """Additive correction to the winner's log-GWP estimate.

    Implements Proposition 1:

        Delta = sigma^2 / sqrt(sigma^2 + tau^2) * a_n

    which grows with the predictive noise sigma (so it is larger where evidence
    is thin) AND with the number of candidates ranked, through a_n. A fixed
    multiplicative band tracks neither; empirical Bayes tracks only the first.

    kappa absorbs model misspecification -- non-normal residuals,
    heteroscedasticity, and error correlation among products from one supplier
    (which makes the effective number of independent draws smaller than n).
    Fit it once on calibration data.
    """
    s = np.sqrt(sigma2 + tau2)
    if s <= 0:
        return 0.0
    return float(kappa * sigma2 / s * expected_min_normal(n))


def rerandomized_select(pred_log, sigma2, gamma=1.0, rng=None):
    """Stage 2', assumption-light alternative to the parametric correction.

    Split the estimate's noise into two independent pieces (Ma, Wang & Wu 2023):

        mu_sel = mu_hat + sqrt(gamma) * xi
        mu_est = mu_hat - xi / sqrt(gamma)          xi ~ N(0, sigma^2)

    Cov(mu_sel, mu_est) = sigma^2 - sigma^2 = 0, so the two are independent
    under the normal model. SELECT on mu_sel and REPORT mu_est. Because the
    reported value played no part in choosing the winner, it is unbiased for
    that winner -- the curse is removed by construction rather than by
    estimating its magnitude.

    Costs variance (mu_est is noisier than mu_hat) but needs no knowledge of
    tau^2, no a_n, and no fitted kappa. gamma trades selection quality against
    report variance; gamma -> 0 recovers naive behaviour.

    Returns (winner_index, reported_log_gwp).
    """
    rng = np.random.default_rng() if rng is None else rng
    p = np.asarray(pred_log, dtype=float)
    xi = rng.normal(0.0, np.sqrt(sigma2), size=p.shape)
    mu_sel = p + np.sqrt(gamma) * xi
    mu_est = p - xi / np.sqrt(gamma)
    w = int(np.argmin(mu_sel))
    return w, float(mu_est[w])


def select_argmin(scores):
    return int(np.argmin(scores))


# --------------------------------------------------------------------------
# certified floor
# --------------------------------------------------------------------------
class SelectionConformal:
    """Distribution-free lower bound on the realized saving of a SELECTED item.

    Calibration must mirror deployment: draw candidate sets, run the same
    selection rule, and record the winner's residual  r = y_true - y_claimed
    (log scale). A high quantile of r bounds how much worse reality can be.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None
        self.n_cal = 0

    def calibrate(self, residuals):
        r = np.asarray(residuals, dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 20:
            raise ValueError(f"need >=20 calibration residuals, got {len(r)}")
        # finite-sample corrected conformal quantile
        n = len(r)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q = float(np.quantile(r, level))
        self.n_cal = n
        return self

    def upper_log_gwp(self, claimed_log):
        """(1-alpha) upper confidence bound on the winner's true log GWP."""
        if self.q is None:
            raise RuntimeError("calibrate() first")
        return np.asarray(claimed_log, dtype=float) + self.q

    def certified_saving(self, baseline_gwp, claimed_log):
        """Lower bound on realized saving (natural scale). May be negative."""
        return baseline_gwp - np.exp(self.upper_log_gwp(claimed_log))


# --------------------------------------------------------------------------
# decision rule
# --------------------------------------------------------------------------
def decide(certified_saving, delta):
    """Recommend the swap only if the CERTIFIED floor clears the margin delta.

    delta is a non-inferiority style margin: the smallest saving worth acting
    on. Returns True to recommend, False to abstain.
    """
    return np.asarray(certified_saving) >= delta


def bh_fdr(pvals, q=0.1):
    """Benjamini-Hochberg: which swap claims survive at FDR level q.

    Used across a bill of materials, where line items sharing suppliers give
    positively dependent p-values -- the regime where BH is valid.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        keep[order[: kmax + 1]] = True
    return keep
