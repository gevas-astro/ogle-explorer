"""Shared numerics for the OGLE explorer: loading, detrending, normalisations, analytic significance."""
import numpy as np

try:
    from scipy.stats import beta as beta_dist
except ImportError:                                                         # K = 1 still works (closed form)
    beta_dist = None

SIDEREAL = 1.0027379
YEAR = 365.25
NORMS = ["standard", "psd", "model", "log", "Bayes log-post", "Bayes post"]
SIGS = ["none", "Baluev", "Beta"]
FAPS = (0.01, 0.001)
RESULT_COLS = ["timestamp", "file", "mjd_start", "mjd_end", "exclude", "N", "nights", "pts_per_night", "T_d",
               "detrend", "clip5sig", "orbit_sub_P", "K", "normalisation", "Pmin", "Pmax",
               "P_best", "f_best", "power_std", "power_norm", "FAP", "FAP_method",
               "sigma_P", "sigma_P_method", "sigma_P_bayes", "P2_over_T", "rms_mmag", "P_fold", "profile_pp_mmag",
               "red_alpha", "red_maxpow99", "red_pvalue"]


# ------------------------------------------------------------------ helpers
def load(path):
    d = np.loadtxt(path)
    return d[:, 0] - 2400000.5, d[:, 1], d[:, 2]


def running_median(t, y, half, min_pts=7):
    """Running median of half-width `half` days; the window is widened to at least `min_pts` points
    (otherwise sparse data give residuals that are exactly zero)."""
    o = np.argsort(t); ts, ys = t[o], y[o]
    out = np.empty_like(ys); lo = hi = 0; n = ts.size
    for i in range(n):
        while ts[lo] < ts[i] - half: lo += 1
        while hi < n and ts[hi] <= ts[i] + half: hi += 1
        a, b = lo, hi
        while b - a < min_pts and (a > 0 or b < n):
            a, b = max(a - 1, 0), min(b + 1, n)
        out[i] = np.median(ys[a:b])
    r = np.empty_like(out); r[o] = out
    return r


def robust_sigma(r):
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def folded_template(t, r, P, nb=20):
    ph = (t / P) % 1
    b = np.minimum((ph * nb).astype(int), nb - 1)
    prof = np.array([np.median(r[b == k]) if (b == k).sum() > 2 else 0.0 for k in range(nb)])
    xc = (np.arange(nb) + 0.5) / nb
    return np.interp(ph, np.r_[xc - 1, xc, xc + 1], np.r_[prof, prof, prof])


def night_id(t):
    return np.floor(t - 0.6)


def season_id(t):
    """Observing season (split at the Magellanic Cloud conjunction; same convention as ogle_tools.season_poly)."""
    return np.floor((t - 55000 + 120) / YEAR)


def parse_exclude(s):
    out = []
    for chunk in s.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.replace(":", "-").split("-")
        out.append((float(a), float(b)))
    return out


def to_norm(p, norm, N, K, chi2ref, pbest):
    """Standard LS power P -> other normalisations (exact for unweighted LS with a floating mean)."""
    p = np.clip(np.asarray(p, float), 0.0, 1 - 1e-15)
    if norm == "standard":
        return p
    if norm == "psd":
        return 0.5 * chi2ref * p
    if norm == "model":
        return p / (1 - p)
    if norm == "log":
        return -np.log1p(-p)
    if norm == "Bayes log-post":
        c = 0.5 * (N - 2 * K - 1)
        return -c * np.log1p(-p) + c * np.log1p(-min(pbest, 1 - 1e-15))
    raise ValueError(norm)


def beta_sf(z, K, N):
    a, b = K, 0.5 * (N - 2 * K - 1)
    if beta_dist is not None:
        return beta_dist.sf(z, a, b)
    if K == 1:
        return (1 - z) ** b
    return np.nan


def beta_isf(q, K, N):
    a, b = K, 0.5 * (N - 2 * K - 1)
    if beta_dist is not None:
        return beta_dist.isf(q, a, b)
    if K == 1:
        return 1 - np.asarray(q) ** (1 / b)
    return np.full(np.shape(q), np.nan)


# ------------------------------------------------------------------ periodogram engines
# Optional fast engine: Toeplitz-LS (pip install "ogle-explorer[fast]"), Krutkowski 2026, arXiv:2611.20300.
# It computes the same multi-harmonic, floating-mean LS power as astropy (standard normalisation) on a uniform
# frequency grid, using NuFFTs and a Toeplitz solve per frequency: ~10x faster for K = 1, ~100x for K = 2-3.
try:
    from toeplitz_ls import tls as _tls
    HAVE_TOEPLITZ = True
except Exception:                                   # not installed, or no wheel for this platform
    _tls = None
    HAVE_TOEPLITZ = False

ENGINES = ["astropy", "Toeplitz"]


def ls_power(t, y, f, nterms=1, engine="astropy"):
    """Standard-normalised (R^2) Lomb-Scargle power with K = nterms harmonics and a floating mean.
    engine='Toeplitz' needs toeplitz_ls and a uniform grid; otherwise (or for unstable bins, returned as NaN by
    Toeplitz-LS) astropy is used, so the result is always complete."""
    from astropy.timeseries import LombScargle
    f = np.asarray(f, float)
    if engine == "Toeplitz" and HAVE_TOEPLITZ and f.size > 2:
        df = (f[-1] - f[0]) / (f.size - 1)
        if df > 0 and np.allclose(np.diff(f), df, rtol=1e-6, atol=0):
            p = np.asarray(_tls.power(f.size, df, f[0], np.asarray(t, float), np.asarray(y, float),
                                      nterms=int(nterms), normalization="standard"), float)
            bad = ~np.isfinite(p)
            if bad.any():                           # ill-conditioned bins: fill them with the astropy value
                p[bad] = LombScargle(t, y, nterms=nterms).power(f[bad])
            return p
    return LombScargle(t, y, nterms=nterms).power(f)
