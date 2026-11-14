"""Helpers for the OGLE BeXRB exercise: loading, detrending, orbital profile, red-noise (Timmer & Koenig)
simulations with the real sampling, a PSRESP-like fit of the noise slope, and alias bookkeeping for NRPs."""
import numpy as np
from astropy.timeseries import LombScargle

SIDEREAL = 1.0027379        # 1/sidereal day  [1/d]
YEAR = 365.25


# ------------------------------------------------------------------ data
def load(path):
    """OGLE XROM .dat: HJD, I, sigma_I, ... -> MJD, I, sigma_I"""
    d = np.loadtxt(path)
    return d[:, 0] - 2400000.5, d[:, 1], d[:, 2]


def select(t, gap=59500.0, part="before", exclude=()):
    """Boolean mask: 'before' / 'after' the COVID gap, minus excluded (outburst) intervals."""
    m = (t < gap) if part == "before" else (t >= gap)
    for a, b in exclude:
        m &= ~((t > a) & (t < b))
    return m


# ------------------------------------------------------------------ detrending
def running_median(t, y, half):
    """Running median of half-width `half` days (O(N) window, t need not be sorted)."""
    o = np.argsort(t); ts, ys = t[o], y[o]
    out = np.empty_like(ys); lo = hi = 0; n = ts.size
    for i in range(n):
        while ts[lo] < ts[i] - half: lo += 1
        while hi < n and ts[hi] <= ts[i] + half: hi += 1
        out[i] = np.median(ys[lo:hi])
    r = np.empty_like(out); r[o] = out
    return r


def season_poly(t, y, deg=2):
    """Subtract a polynomial of degree `deg` per observing season (seasons split at the SMC/LMC conjunction)."""
    s = np.floor((t - 55000 + 120) / YEAR); r = np.empty_like(y)
    for k in np.unique(s):
        m = s == k
        if m.sum() > deg + 3:
            x = t[m] - t[m].mean()
            r[m] = y[m] - np.polyval(np.polyfit(x, y[m], deg), x)
        else:
            r[m] = y[m] - y[m].mean()
    return r


DETRENDS = {
    "median ±50 d": lambda t, y: y - running_median(t, y, 50.0),
    "median ±20 d": lambda t, y: y - running_median(t, y, 20.0),
    "season quadratic": lambda t, y: season_poly(t, y, 2),
}


def robust_sigma(r):
    return 1.4826 * np.median(np.abs(r - np.median(r)))


# ------------------------------------------------------------------ orbital period
def best_period(t, r, P0, frac=0.06, nterms=3, n=3000):
    """Peak of a multi-harmonic LS within P0*(1 +- frac). Returns P, freq grid, power."""
    f = np.linspace(1 / (P0 * (1 + frac)), 1 / (P0 * (1 - frac)), n)
    ls = LombScargle(t, r, nterms=nterms)
    p = ls.power(f)
    i = p.argmax(); df = f[1] - f[0]
    ff = np.linspace(f[i] - df, f[i] + df, 101)               # refine the peak below the grid step
    return 1 / ff[ls.power(ff).argmax()], f, p


def orbital_profile(t, r, P, nb=20, t0=0.0):
    """Median folded profile in nb bins and a periodic interpolator template(t)."""
    ph = ((t - t0) / P) % 1
    b = np.minimum((ph * nb).astype(int), nb - 1)
    prof = np.array([np.median(r[b == k]) if (b == k).sum() > 2 else 0.0 for k in range(nb)])
    prof -= np.median(prof)
    xc = (np.arange(nb) + 0.5) / nb
    xx, pp = np.r_[xc - 1, xc, xc + 1], np.r_[prof, prof, prof]
    return prof, (lambda tt: np.interp(((tt - t0) / P) % 1, xx, pp))


# ------------------------------------------------------------------ red noise with the real sampling
def bpl_psd(alpha, f_low=1 / 2000.0):
    """Power law f^-alpha, flattened below f_low (avoids infinite power at f -> 0)."""
    return lambda f: 1.0 / (1.0 + (f / f_low) ** alpha)


def tk95_sampled(t, alpha, rms, rng, dt=0.02, f_low=1 / 2000.0):
    """Timmer & Koenig light curve on a fine grid (dt) spanning 2x the data, read off at the times t."""
    T = np.ptp(t)
    n = int(2 * T / dt) + 2
    n += n % 2
    f = np.fft.rfftfreq(n, dt)[1:]
    S = bpl_psd(alpha, f_low)(f)
    z = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) * np.sqrt(S / 2)
    z[-1] = z[-1].real
    x = np.fft.irfft(np.r_[0.0, z], n)
    s = rng.integers(0, n // 2)                          # random start: no wrap-around periodicity
    xs = x[s + np.round((t - t.min()) / dt).astype(int)]
    return (xs - xs.mean()) / xs.std() * rms


def binned_logpsd(t, r, fmin, fmax, nbin=12):
    """Mean log10 LS power in log-spaced frequency bins (the 'shape' that PSRESP compares)."""
    edges = np.logspace(np.log10(fmin), np.log10(fmax), nbin + 1)
    f = np.logspace(np.log10(fmin), np.log10(fmax), 3000)
    p = LombScargle(t, r, normalization="psd").power(f)
    idx = np.digitize(f, edges) - 1
    return np.sqrt(edges[1:] * edges[:-1]), np.array([np.mean(np.log10(p[idx == k])) for k in range(nbin)])


def fit_red_noise(t, noise, sig_white, alphas=np.arange(0.5, 3.51, 0.25), nsim=20, rng=None, fmax=0.5):
    """PSRESP-lite: for each slope alpha, simulate (red with the right total rms + white) at the real
    sampling, compare the binned log periodogram with the data's. Returns best alpha and the distance curve."""
    rng = np.random.default_rng(0) if rng is None else rng
    fmin = 2 / np.ptp(t)
    fc, data = binned_logpsd(t, noise, fmin, fmax)
    rms_red = np.sqrt(max(noise.var() - sig_white**2, 1e-12))
    dist = []
    for a in alphas:
        sims = np.array([binned_logpsd(t, tk95_sampled(t, a, rms_red, rng) + rng.normal(0, sig_white, t.size),
                                       fmin, fmax)[1] for _ in range(nsim)])
        mu, sd = sims.mean(0), sims.std(0) + 1e-3
        dist.append(np.sum(((data - mu) / sd) ** 2))
    dist = np.array(dist)
    return float(alphas[dist.argmin()]), alphas, dist, rms_red


def white_level(t, r, dmax=0.3):
    """White-noise sigma from differences of points closer than dmax days (needs several points per night).
    Falls back to None if there are too few pairs."""
    o = np.argsort(t); dt = np.diff(t[o]); dr = np.diff(r[o])
    m = dt < dmax
    return robust_sigma(dr[m]) / np.sqrt(2) if m.sum() > 50 else None


# ------------------------------------------------------------------ NRP / aliases
def alias_family(f0, fmax=4.0, k_max=4, fd=SIDEREAL):
    """All |k*fd +- f0| in (0.05, fmax): the daily-alias family of f0."""
    c = sorted({round(abs(k * fd + s * f0), 4) for k in range(-k_max, k_max + 1) for s in (1, -1)})
    return [x for x in c if 0.05 < x < fmax]


def sine_chi2(t, r, f):
    X = np.c_[np.ones_like(t), np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
    c = np.linalg.lstsq(X, r, rcond=None)[0]
    return float(((r - X @ c) ** 2).sum()), float(np.hypot(c[1], c[2]))


def spectral_window(t, f):
    """|sum exp(2 pi i f t)|^2 / N^2 : 1 at f=0; values near 1 at f = 1/d mean strong daily aliases."""
    ph = 2 * np.pi * np.outer(np.atleast_1d(f), t - t.min())
    return (np.cos(ph).sum(1) ** 2 + np.sin(ph).sum(1) ** 2) / t.size**2


def top_peaks(f, p, k=5, sep=0.01):
    out = []
    for i in np.argsort(p)[::-1]:
        if all(abs(f[i] - f[j]) > sep for j in out):
            out.append(i)
        if len(out) == k:
            break
    return [(float(f[i]), float(p[i])) for i in out]
