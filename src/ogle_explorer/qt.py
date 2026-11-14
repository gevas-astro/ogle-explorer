"""Interactive explorer for OGLE light curves: Qt version (PySide6 + pyqtgraph).

    ogle-explorer                         # pick a *.dat file from the list
    ogle-explorer rxj0520.dat --exclude "60300-60470" --alias auto
    ogle-explorer smc_x3.dat --pmin 0.1 --pmax 3 --porb 44.9

Same analysis as the matplotlib version (ogle-explorer-mpl); the numerics live in core.py and tools.py.
Only the user interface is different:
  * real Qt widgets: typing, spin boxes and sliders never redraw the plots;
  * fast plotting: mouse wheel = zoom, right-drag = zoom one axis, left-drag = pan, right-click = menu
    (incl. "View all" and image export); the 'A' button in a plot corner restores autoscale;
  * slow jobs (periodogram, bootstraps, red-noise sims, window) run in a background thread with a progress
    bar and a Cancel button, so the window stays responsive.

Plots
  Light curve   drag the orange band (or its edges) to choose the interval; grey bands = excluded intervals.
  Detrended     the selected, detrended light curve that goes into the periodogram.
  Periodogram   left-click to choose the fold period. Grey band = alias region (not searched).
                FAP_W = analytic white-noise levels, FAP_R 99 = 99% of the max power in red-noise sims,
                win = spectral window W(f), win_c = window centred on the peak (alias pattern of the peak).
  Folded        fold at the chosen period (default: best period) with binned medians.
  Local posterior (Bayes normalisations only) p(P|D) on a fine grid around the peak.

Install:  pip install .   (from the repository folder)
"""
import argparse
import csv
import glob
import os
import sys
import time
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.exporters                              # noqa: F401  (registers ImageExporter)
from astropy.timeseries import LombScargle
from astropy.table import Table

from . import tools as ot
from .core import (load, running_median, robust_sigma, folded_template, night_id, season_id, parse_exclude,
                   to_norm, beta_sf, beta_isf, SIDEREAL, NORMS, SIGS, FAPS, RESULT_COLS)

pg.setConfigOptions(background="w", foreground="k", antialias=False)

C_SEL = (255, 140, 0, 45)          # orange: selected interval
C_EXC = (120, 120, 120, 80)        # grey: excluded intervals
C_ALIAS = (128, 128, 128, 55)      # grey: alias region
C_SIG = (0, 150, 0, 70)            # green: sigma_P band
BLUE, RED, MAG, GREEN, ORANGE = "#1f77b4", "#d62728", "#c000c0", "#2ca02c", "#ff7f0e"


# =================================================================== analysis core (no GUI)
class Cancelled(Exception):
    pass


class Core:
    """All numerics, as plain functions of a settings dict and the cached analysis. `job` (optional) is the
    background Job, used for progress reports and cancellation."""
    MAXF = 40000

    @staticmethod
    def _report(job, i, n, msg):
        if job is not None:
            job.report(i, n, msg)

    def compute(self, t, y, e, s, job=None):
        self._report(job, 0, 0, "detrending …")
        m = (t >= s["sel"][0]) & (t <= s["sel"][1])
        for a, b in s["exclude"]:
            m &= ~((t > a) & (t < b))
        t, y, e = t[m], y[m], e[m]
        if t.size < 20:
            raise ValueError(f"only {t.size} points in the selection")
        r = y - running_median(t, y, s["half"]) if s["detrend"] else y - np.median(y)
        if s["orbit"]:
            r = r - folded_template(t, r, s["orbit"])
        if s["clip"]:
            k = np.abs(r - np.median(r)) < 5 * robust_sigma(r)
            t, r, e = t[k], r[k], e[k]
        pmin, pmax, K, pcut = s["pmin"], s["pmax"], s["K"], s["pcut"]
        T = np.ptp(t); N = t.size
        nf = int(min(self.MAXF, 5 * T * (1 / pmin - 1 / pmax)))
        f = np.linspace(1 / pmax, 1 / pmin, max(nf, 500))
        self._report(job, 0, 0, f"Lomb-Scargle, K = {K}, {f.size} frequencies …")
        ls = LombScargle(t, r, nterms=K)
        p = ls.power(f)
        search = np.ones(f.size, bool) if pcut is None else (1 / f >= pcut)
        if search.sum() < 10:
            raise ValueError(f"alias cut {pcut:g} d leaves no periods to search")
        ib = np.flatnonzero(search)[p[search].argmax()]; df = f[1] - f[0]
        ff = np.linspace(f[ib] - df, min(f[ib] + df, 1 / pcut if pcut else np.inf), 101)
        pf = ls.power(ff)
        c = dict(t=t, r=r, e=e, f=f, p=p, K=K, N=N, T=T, ls=ls, pmin=pmin, pmax=pmax, pcut=pcut,
                 search=search, fs=f[search], fbest=float(ff[pf.argmax()]), pbest=float(pf.max()),
                 chi2ref=float(np.sum((r - r.mean()) ** 2)), settings=s)
        self._report(job, 0, 0, "local posterior …")
        c["bayes"] = self.bayes_local(c)
        return c

    @staticmethod
    def bayes_local(c):
        """Posterior p(f|D) on a dense grid of ±0.5/T around the peak (the global grid is far coarser than the
        posterior width when N is large)."""
        T = c["T"]
        ff = np.linspace(c["fbest"] - 0.5 / T, c["fbest"] + 0.5 / T, 4001)
        pl = c["ls"].power(ff)
        L = to_norm(pl, "Bayes log-post", c["N"], c["K"], c["chi2ref"], pl.max())
        w = np.exp(L); w /= w.sum()
        fm = float((w * ff).sum()); sf = float(np.sqrt((w * (ff - fm) ** 2).sum()))
        return dict(ff=ff, dens=w / (ff[1] - ff[0]), fmean=fm, sf=sf, sP=sf / fm ** 2)

    @staticmethod
    def analytic(c, meth):
        """(levels in standard power for FAPS, FAP of the peak, label) or None. White noise only."""
        if meth == "none" or c is None:
            return None
        f, K, N, T = c["fs"], c["K"], c["N"], c["T"]
        if meth == "Baluev" and K == 1:
            lev = np.atleast_1d(c["ls"].false_alarm_level(list(FAPS), minimum_frequency=f[0],
                                                          maximum_frequency=f[-1], method="baluev"))
            fap = float(c["ls"].false_alarm_probability(c["pbest"], minimum_frequency=f[0],
                                                        maximum_frequency=f[-1], method="baluev"))
            return lev, fap, "Baluev"
        M = max(T * (f[-1] - f[0]), 1.0)
        q = -np.expm1(np.log1p(-np.array(FAPS)) / M)
        lev = beta_isf(q, K, N)
        tail = beta_sf(c["pbest"], K, N)
        fap = float(-np.expm1(M * np.log1p(-tail))) if np.isfinite(tail) else np.nan
        return lev, fap, f"Beta, M≈{M:.0f}" + (" (Baluev needs K=1)" if meth == "Baluev" else "")

    def red_noise_null(self, c, alpha_txt, nsim, job=None):
        t, r, f, K, s = c["t"], c["r"], c["fs"], c["K"], c["settings"]
        resid = r - c["ls"].model(t, c["fbest"])
        sw = ot.white_level(t, resid)
        if sw is None:
            sw = float(np.median(c["e"]))
        sw = min(sw, 0.95 * resid.std())
        rms_red = float(np.sqrt(max(resid.var() - sw ** 2, 1e-12)))
        txt = alpha_txt.strip().lower()
        if txt in ("", "fit"):
            self._report(job, 0, 0, "fitting the red-noise slope (PSRESP-lite) …")
            alpha = ot.fit_red_noise(t, resid, sw, alphas=np.arange(0.5, 3.51, 0.5), nsim=10,
                                     rng=np.random.default_rng(3))[0]
            how = "fitted"
        else:
            alpha = float(txt); how = "set"
        rng = np.random.default_rng(1)
        maxp = np.empty(nsim)
        for i in range(nsim):
            self._report(job, i, nsim, f"red-noise sims (α = {alpha:g})")
            x = ot.tk95_sampled(t, alpha, 1.0, rng)
            if s["detrend"]:
                x = x - running_median(t, x, s["half"])
            x = (x - x.mean()) / x.std() * rms_red + rng.normal(0, sw, t.size)
            maxp[i] = LombScargle(t, x, nterms=K).power(f).max()
        return dict(alpha=alpha, how=how, rms_red=rms_red, sw=sw, nsim=nsim, maxp=maxp,
                    lev95=float(np.percentile(maxp, 95)), lev99=float(np.percentile(maxp, 99)),
                    pval=float((np.sum(maxp >= c["pbest"]) + 1) / (nsim + 1)))

    @staticmethod
    def _peak(c, t, r, f0):
        K, T = c["K"], c["T"]
        ls = LombScargle(t, r, nterms=K)
        ff = np.linspace(f0 - 3 / T, f0 + 3 / T, 241); ff = ff[ff > 0]
        i = ls.power(ff).argmax(); d = ff[1] - ff[0]
        f2 = np.linspace(ff[i] - d, ff[i] + d, 41)
        return 1 / f2[ls.power(f2).argmax()]

    def sigma_P(self, c, kind, nboot, job=None, label=""):
        t, r, f0 = c["t"], c["r"], c["fbest"]
        rng = np.random.default_rng(42)
        if kind == "point":
            Ps = []
            for i in range(nboot):
                self._report(job, i, nboot, label or "point bootstrap")
                k = rng.integers(0, t.size, t.size)
                Ps.append(self._peak(c, t[k], r[k], f0))
            return float(np.std(Ps, ddof=1)), t.size
        g = night_id(t) if kind == "night" else season_id(t)
        ug, inv = np.unique(g, return_inverse=True)
        blocks = [np.flatnonzero(inv == k) for k in range(ug.size)]
        n = ug.size
        if n < 3:
            return np.nan, n
        if kind == "jackknife":
            Ps = []
            for k in range(n):
                self._report(job, k, n, label or "season jackknife")
                Ps.append(self._peak(c, t[inv != k], r[inv != k], f0))
            Ps = np.array(Ps)
            return float(np.sqrt((n - 1) / n * np.sum((Ps - Ps.mean()) ** 2))), n
        Ps = []
        for i in range(nboot):
            self._report(job, i, nboot, label or f"{kind} bootstrap")
            k = np.concatenate([blocks[j] for j in rng.integers(0, n, n)])
            Ps.append(self._peak(c, t[k], r[k], f0))
        return float(np.std(Ps, ddof=1)), n

    @staticmethod
    def sigma_analytic(c):
        """Montgomery & O'Donoghue (1999), white noise, single sine."""
        chi2, A = ot.sine_chi2(c["t"], c["r"], c["fbest"])
        sig = np.sqrt(chi2 / (c["N"] - 3))
        return float(np.sqrt(6 / c["N"]) * sig / (np.pi * c["T"] * A) / c["fbest"] ** 2)

    def boot_compare(self, c, nboot, job=None):
        rows = []
        for kind, name in [("point", "point boot"), ("night", "night boot"), ("season", "season boot"),
                           ("jackknife", "season jackknife")]:
            s, n = self.sigma_P(c, kind, nboot, job, label=f"boot compare: {name}")
            rows.append((name, s, n))
        rows.append(("analytic white", self.sigma_analytic(c), c["N"]))
        return rows

    def windows(self, c, which, job=None):
        """Spectral window W(f) ('win') or the peak-centred alias pattern max[W(f-fp), W(f+fp)] ('win_c')."""
        f, t = c["f"], c["t"]
        shifts = [0.0] if which == "win" else [c["fbest"], -c["fbest"]]
        out = np.zeros(f.size)
        n = f.size; step = 2000; tot = len(shifts) * ((n + step - 1) // step); k = 0
        for sh in shifts:
            w = np.empty(n)
            for i in range(0, n, step):
                self._report(job, k, tot, f"spectral window ({which})"); k += 1
                w[i:i + step] = ot.spectral_window(t, f[i:i + step] - sh)
            out = np.maximum(out, w)
        return out


# =================================================================== background jobs
class Job(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, name, fn, parent=None):
        super().__init__(parent)
        self.name, self.fn, self.cancel = name, fn, False

    def report(self, i, n, msg):
        if self.cancel:
            raise Cancelled()
        self.progress.emit(i, n, msg)

    def run(self):
        try:
            res = self.fn(self)
        except Cancelled:
            self.failed.emit("cancelled")
        except Exception as err:                                        # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(err) or err.__class__.__name__)
        else:
            self.done.emit(res)


# =================================================================== GUI
class PeriodAxis(pg.AxisItem):
    """Bottom axis that labels a log axis with plain numbers at 1, 2, 5 x 10^n (pyqtgraph's default crowds)."""

    def tickStrings(self, values, scale, spacing):
        if not self.logMode:
            return super().tickStrings(values, scale, spacing)
        out = []
        for v in values:
            x = 10 ** v
            m = x / 10 ** np.floor(np.log10(x) + 1e-9)
            out.append(f"{x:g}" if min(abs(m - k) for k in (1, 2, 5)) < 1e-6 else "")
        return out


def plot_widget(title=None, xlabel=None, ylabel=None, period_axis=False):
    w = pg.PlotWidget(axisItems={"bottom": PeriodAxis(orientation="bottom")} if period_axis else None)
    pi = w.getPlotItem()
    for ax in ("left", "bottom"):
        pi.getAxis(ax).enableAutoSIPrefix(False)
    if title:
        pi.setTitle(title, size="9pt")
    if xlabel:
        pi.setLabel("bottom", xlabel)
    if ylabel:
        pi.setLabel("left", ylabel)
    pi.showGrid(x=False, y=False)
    return w


def line_edit(text="", width=None, tip=None, placeholder=None):
    e = QtWidgets.QLineEdit(text)
    if width:
        e.setMaximumWidth(width)
    if tip:
        e.setToolTip(tip)
    if placeholder:
        e.setPlaceholderText(placeholder)
    return e


class ExplorerQt(QtWidgets.QMainWindow):
    def __init__(self, path=None, pmin=10.0, pmax=60.0, porb=None, workdir=".", outdir="explorer_output",
                 live=False, exclude="", alias=""):
        super().__init__()
        self.core = Core()
        self.workdir, self.outdir = workdir, outdir
        self.t = self.y = self.e = np.array([])
        self.name = ""
        self.sel = (0.0, 1.0)
        self.cache = None
        self.cache_id = 0
        self.Pfold = None
        self.unc = self.boot = self.red = None
        self.win = {}                                   # 'win' / 'win_c' -> array
        self.stale = False
        self.job = None
        self.pending_update = False
        self.ls_log = False

        self.setWindowTitle("OGLE explorer (Qt)")
        self.resize(1650, 1000)
        self._build_plots()
        self._build_controls(pmin, pmax, porb, live, exclude, alias)
        self._build_statusbar()

        self.live_timer = QtCore.QTimer(self, singleShot=True, interval=350)
        self.live_timer.timeout.connect(self.update_analysis)

        self.refresh_file_list()
        if path is None and self.files:
            path = self.files[0]
        if path:
            self.load_file(path)
        else:
            self.set_status("no *.dat file here: use Browse… or type a path")

    # ------------------------------------------------------------------ layout
    def _build_plots(self):
        self.p_lc = plot_widget(None, "MJD", "I [mag]")
        self.p_dt = plot_widget("detrended selection", "MJD", "ΔI [mag]")
        self.p_ls = plot_widget("periodogram: left-click to choose the fold period", "P [d]", "LS power",
                                period_axis=True)
        self.p_fd = plot_widget("folded", "phase", "ΔI [mag]")
        self.p_post = plot_widget("local posterior p(P|D)", "P [d]")
        for p in (self.p_lc, self.p_dt, self.p_fd):
            p.getPlotItem().invertY(True)
        self.p_post.getPlotItem().hideAxis("left")
        self.p_post.setMinimumHeight(170)

        self.lc_curve = self.p_lc.plot([], [], pen=None, symbol="o", symbolSize=2, symbolPen=None,
                                       symbolBrush="k")
        self.region = pg.LinearRegionItem(values=(0, 1), brush=C_SEL, hoverBrush=(255, 140, 0, 80))
        self.region.setZValue(10)
        self.region.sigRegionChangeFinished.connect(self.on_region)
        self.p_lc.addItem(self.region)
        self.exc_items = []
        self.ls_legend = self.p_ls.getPlotItem().addLegend(offset=(-8, 8), labelTextSize="8pt",
                                                           verSpacing=-3, colCount=2)
        self.p_ls.scene().sigMouseClicked.connect(self.on_ls_click)

        self.summary = QtWidgets.QPlainTextEdit(readOnly=True)
        f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont); f.setPointSize(9)
        self.summary.setFont(f)
        self.summary.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

        right_bottom = QtWidgets.QSplitter(Qt.Vertical)
        right_bottom.addWidget(self.summary)
        right_bottom.addWidget(self.p_post)
        right_bottom.setSizes([300, 180])

        mid = QtWidgets.QSplitter(Qt.Horizontal); mid.addWidget(self.p_dt); mid.addWidget(self.p_ls)
        bot = QtWidgets.QSplitter(Qt.Horizontal); bot.addWidget(self.p_fd); bot.addWidget(right_bottom)
        for s in (mid, bot):
            s.setSizes([600, 600])
        plots = QtWidgets.QSplitter(Qt.Vertical)
        for w in (self.p_lc, mid, bot):
            plots.addWidget(w)
        plots.setSizes([260, 360, 360])
        self.plots_splitter = plots

        self.controls = QtWidgets.QWidget()
        scroll = QtWidgets.QScrollArea(); scroll.setWidget(self.controls); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330); scroll.setMaximumWidth(400)
        main = QtWidgets.QSplitter(Qt.Horizontal)
        main.addWidget(plots); main.addWidget(scroll)
        main.setStretchFactor(0, 1); main.setStretchFactor(1, 0)
        self.setCentralWidget(main)

    def _build_controls(self, pmin, pmax, porb, live, exclude, alias):
        V = QtWidgets.QVBoxLayout(self.controls)
        V.setContentsMargins(6, 6, 6, 6)

        # --- data
        g = QtWidgets.QGroupBox("data"); L = QtWidgets.QGridLayout(g)
        self.cb_file = QtWidgets.QComboBox(); self.cb_file.activated.connect(self.on_file_combo)
        b_browse = QtWidgets.QPushButton("Browse…"); b_browse.clicked.connect(self.on_browse)
        self.le_path = line_edit(placeholder="path to a .dat file, Enter to load")
        self.le_path.returnPressed.connect(lambda: self.load_file(self.le_path.text().strip()))
        b_full = QtWidgets.QPushButton("full range"); b_full.clicked.connect(self.on_full_range)
        self.le_exc = line_edit(exclude, tip="MJD intervals to leave out, e.g. 56550-56800, 60300-60470",
                                placeholder="e.g. 56550-56800, 60300-60470")
        L.addWidget(self.cb_file, 0, 0, 1, 2); L.addWidget(b_browse, 0, 2)
        L.addWidget(self.le_path, 1, 0, 1, 3)
        L.addWidget(QtWidgets.QLabel("exclude"), 2, 0); L.addWidget(self.le_exc, 2, 1, 1, 2)
        L.addWidget(b_full, 3, 2)
        self.lb_sel = QtWidgets.QLabel(""); L.addWidget(self.lb_sel, 3, 0, 1, 2)
        V.addWidget(g)

        # --- preparation
        g = QtWidgets.QGroupBox("preparation"); L = QtWidgets.QGridLayout(g)
        self.ck_det = QtWidgets.QCheckBox("detrend ±[d]"); self.ck_det.setChecked(True)
        self.sl_det = QtWidgets.QSlider(Qt.Horizontal); self.sl_det.setRange(2, 200)      # half-days
        self.sp_det = QtWidgets.QDoubleSpinBox(); self.sp_det.setRange(1, 100); self.sp_det.setSingleStep(0.5)
        self.sp_det.setDecimals(1); self.sp_det.setKeyboardTracking(False)
        self.sl_det.setValue(100); self.sp_det.setValue(50)
        self.ck_orb = QtWidgets.QCheckBox("subtract orbit, Porb [d]")
        self.le_orb = line_edit("" if porb is None else f"{porb:g}", 80)
        self.ck_clip = QtWidgets.QCheckBox("clip 5σ"); self.ck_clip.setChecked(True)
        L.addWidget(self.ck_det, 0, 0); L.addWidget(self.sl_det, 0, 1); L.addWidget(self.sp_det, 0, 2)
        L.addWidget(self.ck_orb, 1, 0, 1, 2); L.addWidget(self.le_orb, 1, 2)
        L.addWidget(self.ck_clip, 2, 0)
        V.addWidget(g)

        # --- periodogram
        g = QtWidgets.QGroupBox("periodogram"); L = QtWidgets.QGridLayout(g)
        self.sl_K = QtWidgets.QSlider(Qt.Horizontal); self.sl_K.setRange(1, 5); self.sl_K.setValue(1)
        self.sl_K.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sp_K = QtWidgets.QSpinBox(); self.sp_K.setRange(1, 5); self.sp_K.setValue(1)
        self.sp_K.setKeyboardTracking(False)
        self.le_pmin = line_edit(f"{pmin:g}", 80); self.le_pmax = line_edit(f"{pmax:g}", 80)
        self.le_alias = line_edit(alias, 80, tip="periods below this are the alias region: shaded and not "
                                  "searched. Empty = off, 'auto' = 2 sidereal days, or a number [d]",
                                  placeholder="auto / d")
        self.co_norm = QtWidgets.QComboBox(); self.co_norm.addItems(NORMS)
        self.co_sig = QtWidgets.QComboBox(); self.co_sig.addItems(SIGS); self.co_sig.setCurrentText("Baluev")
        self.co_sig.setToolTip("analytic significance, valid for WHITE noise only")
        self.ck_win = QtWidgets.QCheckBox("window (win)")
        self.ck_winc = QtWidgets.QCheckBox("window @peak (win_c)")
        L.addWidget(QtWidgets.QLabel("harmonics K"), 0, 0); L.addWidget(self.sl_K, 0, 1)
        L.addWidget(self.sp_K, 0, 2)
        L.addWidget(QtWidgets.QLabel("Pmin [d]"), 1, 0); L.addWidget(self.le_pmin, 1, 1)
        L.addWidget(QtWidgets.QLabel("Pmax [d]"), 2, 0); L.addWidget(self.le_pmax, 2, 1)
        L.addWidget(QtWidgets.QLabel("alias P <"), 3, 0); L.addWidget(self.le_alias, 3, 1)
        L.addWidget(QtWidgets.QLabel("normalisation"), 4, 0); L.addWidget(self.co_norm, 4, 1, 1, 2)
        L.addWidget(QtWidgets.QLabel("analytic sig."), 5, 0); L.addWidget(self.co_sig, 5, 1, 1, 2)
        L.addWidget(self.ck_win, 6, 0, 1, 2); L.addWidget(self.ck_winc, 7, 0, 1, 2)
        V.addWidget(g)

        # --- update
        h = QtWidgets.QHBoxLayout()
        self.ck_live = QtWidgets.QCheckBox("live update"); self.ck_live.setChecked(live)
        self.b_update = QtWidgets.QPushButton("Update"); self.b_update.setMinimumHeight(34)
        f = self.b_update.font(); f.setBold(True); self.b_update.setFont(f)
        self.b_update.setShortcut(QtGui.QKeySequence("Ctrl+Return"))
        self.b_update.setToolTip("recompute (Ctrl+Enter)")
        h.addWidget(self.ck_live); h.addWidget(self.b_update, 1)
        V.addLayout(h)

        # --- uncertainty and significance
        g = QtWidgets.QGroupBox("uncertainty and significance (background)"); L = QtWidgets.QGridLayout(g)
        self.le_alpha = line_edit("fit", 70, tip="red-noise PSD slope α (P ∝ f^-α): a number, or 'fit' "
                                  "= PSRESP-lite fit over 0.5…3.5 (~30 s)")
        self.sp_nsim = QtWidgets.QSpinBox(); self.sp_nsim.setRange(5, 10000); self.sp_nsim.setValue(50)
        self.sp_nboot = QtWidgets.QSpinBox(); self.sp_nboot.setRange(10, 10000); self.sp_nboot.setValue(200)
        self.b_jk = QtWidgets.QPushButton("season jackknife")
        self.b_sb = QtWidgets.QPushButton("season boot")
        self.b_cmp = QtWidgets.QPushButton("boot compare")
        self.b_red = QtWidgets.QPushButton("red-noise null")
        L.addWidget(QtWidgets.QLabel("red α"), 0, 0); L.addWidget(self.le_alpha, 0, 1)
        L.addWidget(QtWidgets.QLabel("N sims"), 1, 0); L.addWidget(self.sp_nsim, 1, 1)
        L.addWidget(QtWidgets.QLabel("N boot"), 2, 0); L.addWidget(self.sp_nboot, 2, 1)
        L.addWidget(self.b_jk, 3, 0); L.addWidget(self.b_sb, 3, 1)
        L.addWidget(self.b_cmp, 4, 0); L.addWidget(self.b_red, 4, 1)
        V.addWidget(g)

        # --- output
        g = QtWidgets.QGroupBox("output"); L = QtWidgets.QGridLayout(g)
        self.le_out = line_edit(self.outdir)
        self.b_panels = QtWidgets.QPushButton("save panels")
        self.b_row = QtWidgets.QPushButton("save row")
        L.addWidget(QtWidgets.QLabel("folder"), 0, 0); L.addWidget(self.le_out, 0, 1)
        L.addWidget(self.b_panels, 1, 0); L.addWidget(self.b_row, 1, 1)
        V.addWidget(g)
        V.addStretch(1)

        # --- signals: analysis parameters -> changed(); display options -> redraw()
        self.sl_det.valueChanged.connect(lambda v: self._sync(self.sp_det, v / 2))
        self.sp_det.valueChanged.connect(lambda v: self._sync(self.sl_det, int(round(2 * v)), True))
        self.sl_K.valueChanged.connect(lambda v: self._sync(self.sp_K, v))
        self.sp_K.valueChanged.connect(lambda v: self._sync(self.sl_K, v, True))
        for w in (self.ck_det, self.ck_orb, self.ck_clip):
            w.toggled.connect(self.changed)
        self.ck_det.toggled.connect(lambda on: (self.sl_det.setEnabled(on), self.sp_det.setEnabled(on)))
        for w in (self.le_pmin, self.le_pmax, self.le_alias, self.le_orb, self.le_exc):
            w.editingFinished.connect(self._edit_finished)
        self.le_exc.editingFinished.connect(self.draw_exclusions)
        self.co_norm.currentIndexChanged.connect(self.redraw)
        self.co_sig.currentIndexChanged.connect(self.redraw)
        self.ck_win.toggled.connect(self.ensure_windows)
        self.ck_winc.toggled.connect(self.ensure_windows)
        self.ck_live.toggled.connect(lambda on: self.update_analysis() if on and self.stale else None)
        self.b_update.clicked.connect(self.update_analysis)
        self.b_jk.clicked.connect(lambda: self.run_sigma("jackknife"))
        self.b_sb.clicked.connect(lambda: self.run_sigma("season"))
        self.b_cmp.clicked.connect(self.run_compare)
        self.b_red.clicked.connect(self.run_red)
        self.b_panels.clicked.connect(self.save_panels)
        self.b_row.clicked.connect(self.save_row)
        self.job_buttons = [self.b_jk, self.b_sb, self.b_cmp, self.b_red]

    def _build_statusbar(self):
        sb = self.statusBar()
        self.lb_status = QtWidgets.QLabel("")
        self.progress = QtWidgets.QProgressBar(); self.progress.setMaximumWidth(260); self.progress.hide()
        self.b_cancel = QtWidgets.QPushButton("Cancel"); self.b_cancel.hide()
        self.b_cancel.clicked.connect(self.cancel_job)
        sb.addWidget(self.lb_status, 1); sb.addPermanentWidget(self.progress); sb.addPermanentWidget(self.b_cancel)

    # ------------------------------------------------------------------ small helpers
    def _sync(self, other, value, from_box=False):
        """Keep slider and box in step without feedback loops; the box is the one that triggers changed()."""
        other.blockSignals(True); other.setValue(value); other.blockSignals(False)
        self.changed()

    def _edit_finished(self):
        w = self.sender()
        if w is not None and w.isModified():
            w.setModified(False)
            self.changed()

    def set_status(self, msg, warn=False):
        self.lb_status.setText(msg)
        self.lb_status.setStyleSheet("color: #c0392b;" if warn else "")

    def mark_stale(self, on):
        self.stale = on
        self.b_update.setStyleSheet("background-color: #f5b041;" if on else "")
        if on:
            self.set_status("settings changed: press Update (Ctrl+Enter)", warn=True)

    def settings(self):
        """Read all analysis parameters from the widgets; raises ValueError with a readable message."""
        try:
            pmin, pmax = float(self.le_pmin.text()), float(self.le_pmax.text())
        except ValueError:
            raise ValueError("bad Pmin / Pmax")
        if not 0 < pmin < pmax:
            raise ValueError("need 0 < Pmin < Pmax")
        try:
            exc = parse_exclude(self.le_exc.text())
        except ValueError:
            raise ValueError("bad 'exclude' (use e.g. 56550-56800, 60300-60470)")
        a = self.le_alias.text().strip().lower()
        if a in ("", "off", "none", "0"):
            pcut = None
        elif a == "auto":
            pcut = 2 / SIDEREAL
        else:
            try:
                pcut = float(a)
            except ValueError:
                raise ValueError("alias P< must be a number, 'auto' or empty")
        try:
            porb = float(self.le_orb.text())
        except ValueError:
            porb = None
        det = self.ck_det.isChecked()
        return dict(sel=self.sel, exclude=exc, exclude_txt=self.le_exc.text().strip(), detrend=det,
                    half=self.sp_det.value(), detrend_txt=f"median ±{self.sp_det.value():g} d" if det else "none",
                    orbit=porb if (self.ck_orb.isChecked() and porb) else None, clip=self.ck_clip.isChecked(),
                    K=self.sp_K.value(), pmin=pmin, pmax=pmax, pcut=pcut)

    # ------------------------------------------------------------------ files and selection
    def refresh_file_list(self):
        self.files = sorted(glob.glob(os.path.join(self.workdir, "*.dat")))
        self.cb_file.clear()
        self.cb_file.addItems([os.path.basename(p) for p in self.files] or ["(no *.dat here)"])

    def on_file_combo(self, i):
        if 0 <= i < len(self.files):
            self.load_file(self.files[i])

    def on_browse(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "OGLE light curve", self.workdir,
                                                     "OGLE data (*.dat *.txt);;all files (*)")
        if p:
            self.load_file(p)

    def load_file(self, path):
        if not path:
            return
        if not os.path.isfile(path):
            self.set_status(f"file not found: {path}", warn=True); return
        try:
            t, y, e = load(path)
        except Exception as err:                                        # noqa: BLE001
            self.set_status(f"cannot read {os.path.basename(path)}: {err}", warn=True); return
        o = np.argsort(t)
        self.t, self.y, self.e = t[o], y[o], e[o]
        self.name = path
        self.le_path.setText(path)
        for i, p in enumerate(self.files):
            if os.path.abspath(p) == os.path.abspath(path):
                self.cb_file.setCurrentIndex(i)
        self.setWindowTitle(f"OGLE explorer (Qt) - {os.path.basename(path)}")
        self.lc_curve.setData(self.t, self.y)
        self.p_lc.getPlotItem().setTitle(f"{os.path.basename(path)}: drag the orange band to choose the "
                                         "interval", size="9pt")
        self.set_selection(self.t.min(), self.t.max())
        self.p_lc.getPlotItem().enableAutoRange()
        self.draw_exclusions()
        self.cache = None; self.Pfold = None
        self.update_analysis()

    def set_selection(self, x0, x1):
        self.sel = (float(x0), float(x1))
        self.region.blockSignals(True); self.region.setRegion(self.sel); self.region.blockSignals(False)
        self.lb_sel.setText(f"MJD {x0:.0f} – {x1:.0f}")

    def on_region(self):
        x0, x1 = self.region.getRegion()
        if abs(x0 - self.sel[0]) < 1e-6 and abs(x1 - self.sel[1]) < 1e-6:
            return
        self.sel = (float(x0), float(x1))
        self.lb_sel.setText(f"MJD {x0:.0f} – {x1:.0f}")
        self.Pfold = None
        self.changed()

    def on_full_range(self):
        if self.t.size:
            self.set_selection(self.t.min(), self.t.max()); self.Pfold = None; self.changed()

    def draw_exclusions(self):
        for it in self.exc_items:
            self.p_lc.removeItem(it)
        self.exc_items = []
        try:
            exc = parse_exclude(self.le_exc.text())
        except ValueError:
            return
        for a, b in exc:
            it = pg.LinearRegionItem(values=(a, b), movable=False, brush=C_EXC, pen=pg.mkPen(None))
            it.setZValue(5)
            self.p_lc.addItem(it); self.exc_items.append(it)

    # ------------------------------------------------------------------ analysis control
    def changed(self, *args):
        if self.t.size == 0:
            return
        if self.ck_live.isChecked():
            self.live_timer.start()                     # debounce: recompute once the user pauses
        elif not self.stale:
            self.mark_stale(True)

    def update_analysis(self):
        if self.t.size == 0:
            return
        if self.job is not None:                        # finish the running job first
            self.pending_update = True
            return
        try:
            s = self.settings()
        except ValueError as err:
            self.set_status(str(err), warn=True); return
        t, y, e = self.t, self.y, self.e
        self.start_job("periodogram", lambda job: self.core.compute(t, y, e, s, job), self.on_computed)

    def on_computed(self, c):
        self.cache = c
        self.cache_id += 1
        if self.Pfold is not None and not c["pmin"] <= self.Pfold <= c["pmax"]:
            self.Pfold = None
        self.unc = self.boot = self.red = None
        self.win = {}
        self.mark_stale(False)
        self.set_status(f"done: best P = {1 / c['fbest']:.6g} d")
        self.redraw()
        self.ensure_windows()

    def start_job(self, name, fn, on_done):
        if self.job is not None:
            self.set_status(f"busy with '{self.job.name}' (Cancel it first)", warn=True); return False
        job = Job(name, fn, self)
        cid = self.cache_id
        job.progress.connect(self.on_progress)
        job.done.connect(lambda res: self._job_done(job, res, on_done, cid, name))
        job.failed.connect(lambda msg: self._job_failed(job, msg))
        self.job = job
        for b in self.job_buttons + [self.b_update]:
            b.setEnabled(False)
        self.progress.setRange(0, 0); self.progress.show(); self.b_cancel.show()
        self.set_status(f"{name} …")
        job.start()
        return True

    def _finish_job(self):
        self.job = None
        for b in self.job_buttons + [self.b_update]:
            b.setEnabled(True)
        self.progress.hide(); self.b_cancel.hide()

    def _job_done(self, job, res, on_done, cid, name):
        self._finish_job()
        if name != "periodogram" and cid != self.cache_id:
            self.set_status(f"{name}: discarded (the analysis changed meanwhile)", warn=True)
        else:
            on_done(res)
        if self.pending_update:
            self.pending_update = False
            self.update_analysis()

    def _job_failed(self, job, msg):
        self._finish_job()
        self.set_status(f"{job.name}: {msg}", warn=True)
        if self.pending_update:
            self.pending_update = False
            self.update_analysis()

    def on_progress(self, i, n, msg):
        if n > 0:
            self.progress.setRange(0, n); self.progress.setValue(i)
            self.set_status(f"{msg}: {i}/{n}")
        else:
            self.progress.setRange(0, 0)
            self.set_status(msg)

    def cancel_job(self):
        if self.job is not None:
            self.job.cancel = True
            self.set_status(f"cancelling {self.job.name} …")

    # jobs ---------------------------------------------------------------
    def ensure_windows(self, *args):
        """Compute every ticked window that is not cached yet (in one background job), then redraw."""
        if self.cache is None:
            return
        need = [k for k, ck in (("win", self.ck_win), ("win_c", self.ck_winc)) if ck.isChecked() and k not in self.win]
        if not need:
            self.redraw(); return
        if self.job is not None:                        # retry when the running job has finished
            QtCore.QTimer.singleShot(300, self.ensure_windows); return
        c = self.cache

        def done(ws):
            self.win.update(ws); self.set_status(""); self.redraw()
        self.start_job("window", lambda job: {k: self.core.windows(c, k, job) for k in need}, done)

    def run_sigma(self, kind):
        if self.cache is None:
            return
        c = self.cache; nb = self.sp_nboot.value()
        name = {"jackknife": "season jackknife", "season": "season boot"}[kind]

        def done(res):
            s, n = res
            if not np.isfinite(s):
                self.set_status(f"{name}: need ≥ 3 seasons in the selection (have {n})", warn=True); return
            self.unc = dict(method=name, sigma=s, n=n)
            self.set_status(f"{name}: σ_P = {s:.4g} d ({n} seasons)")
            self.redraw()
        self.start_job(name, lambda job: self.core.sigma_P(c, kind, nb, job), done)

    def run_compare(self):
        if self.cache is None:
            return
        c = self.cache; nb = self.sp_nboot.value()

        def done(rows):
            self.boot = rows
            if np.isfinite(rows[2][1]):
                self.unc = dict(method="season boot", sigma=rows[2][1], n=rows[2][2])
            self.set_status("boot compare done")
            self.redraw()
        self.start_job("boot compare", lambda job: self.core.boot_compare(c, nb, job), done)

    def run_red(self):
        if self.cache is None:
            return
        c = self.cache; txt = self.le_alpha.text(); nsim = self.sp_nsim.value()
        if txt.strip().lower() not in ("", "fit"):
            try:
                float(txt)
            except ValueError:
                self.set_status("red α must be a number or 'fit'", warn=True); return

        def done(res):
            self.red = res
            self.set_status(f"red-noise null: α = {res['alpha']:g} ({res['how']}), p = {res['pval']:.2g}")
            self.redraw()
        self.start_job("red-noise null", lambda job: self.core.red_noise_null(c, txt, nsim, job), done)

    # ------------------------------------------------------------------ drawing
    def norm(self):
        return self.co_norm.currentText()

    def nrm(self, p):
        c = self.cache
        return float(to_norm(p, self.norm(), c["N"], c["K"], c["chi2ref"], c["pbest"]))

    def display(self):
        c = self.cache; norm = self.norm()
        if norm == "Bayes post":
            L = to_norm(c["p"], "Bayes log-post", c["N"], c["K"], c["chi2ref"], c["pbest"])
            w = np.exp(L)
            return w / (w.sum() * (c["f"][1] - c["f"][0])), "p(f | D)  [d]"
        lab = {"standard": "LS power", "psd": "LS power (psd)", "model": "LS power (model)",
               "log": "LS power (log)", "Bayes log-post": "ln p(f|D) − max"}[norm]
        return to_norm(c["p"], norm, c["N"], c["K"], c["chi2ref"], c["pbest"]), f"{lab}, K = {c['K']}"

    def fold_data(self):
        c = self.cache; t, r = c["t"], c["r"]
        P = self.Pfold or 1 / c["fbest"]
        ph = ((t - t.min()) / P) % 1
        nb = 20; b = np.minimum((ph * nb).astype(int), nb - 1)
        med = np.array([np.median(r[b == k]) if (b == k).any() else np.nan for k in range(nb)])
        return P, ph, (np.arange(nb) + 0.5) / nb, med

    def redraw(self, *args):
        for p in (self.p_dt, self.p_ls, self.p_fd, self.p_post):
            p.getPlotItem().clear()
        self.ls_legend.clear()
        if self.cache is None:
            self.summary.setPlainText(""); return
        self.draw_dt(); self.draw_ls(); self.draw_fold(); self.draw_post()
        L, R, _ = self.summary_text()
        self.summary.setPlainText("\n".join(L) + ("\n\n" + "\n".join(R) if R else ""))

    def draw_dt(self):
        c = self.cache; s = c["settings"]
        self.p_dt.plot(c["t"], c["r"], pen=None, symbol="o", symbolSize=2, symbolPen=None, symbolBrush="k")
        extra = (", orbit removed" if s["orbit"] else "") + (", 5σ clip" if s["clip"] else "")
        self.p_dt.getPlotItem().setTitle(f"detrend: {s['detrend_txt']}{extra};  N = {c['N']}", size="9pt")
        self.p_dt.getPlotItem().enableAutoRange()

    def draw_ls(self):
        c = self.cache; f = c["f"]; Pg = 1 / f; norm = self.norm(); pi = self.p_ls.getPlotItem()
        self.ls_log = c["pmax"] / c["pmin"] > 20
        pi.setLogMode(x=self.ls_log, y=False)
        px = np.log10 if self.ls_log else (lambda v: v)
        y, ylab = self.display()
        pi.setLabel("left", ylab)
        x0, x1 = Pg.min(), Pg.max()
        if c["pcut"] is not None and c["pcut"] > c["pmin"]:
            reg = pg.LinearRegionItem(values=(px(c["pmin"]), px(min(c["pcut"], c["pmax"]))), movable=False,
                                      brush=C_ALIAS, pen=pg.mkPen(None))
            reg.setZValue(-10); pi.addItem(reg)
            self.ls_legend.addItem(pg.PlotDataItem(pen=None, symbol="s", symbolSize=10, symbolPen=None,
                                                   symbolBrush=C_ALIAS[:3] + (110,)), "alias")
        pi.plot(Pg, y, pen=pg.mkPen("k", width=1))
        if norm != "Bayes post":
            an = self.core.analytic(c, self.co_sig.currentText())
            if an is not None and np.all(np.isfinite(an[0])):
                for lev, fap, style in zip(an[0], FAPS, (Qt.DashLine, Qt.DotLine)):
                    pi.plot([x0, x1], [self.nrm(lev)] * 2, pen=pg.mkPen(BLUE, width=1.2, style=style),
                            name=f"FAP_W {100 * (1 - fap):g}")
            if self.red is not None:
                pi.plot([x0, x1], [self.nrm(self.red["lev99"])] * 2, pen=pg.mkPen(RED, width=1.4), name="FAP_R 99")
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        if norm == "Bayes log-post":
            lo = max(lo, -60.0)
        for key, ck, col in (("win", self.ck_win, MAG), ("win_c", self.ck_winc, RED)):
            if ck.isChecked() and key in self.win:
                pi.plot(Pg, lo - self.win[key] * (hi - lo), pen=pg.mkPen(col, width=1), name=key)
        Pb = 1 / c["fbest"]; P = self.Pfold or Pb
        if self.unc is not None:
            s = self.unc["sigma"]
            band = pg.LinearRegionItem(values=(px(Pb - s), px(Pb + s)), movable=False, brush=C_SIG,
                                       pen=pg.mkPen(None))
            pi.addItem(band)
            pi.plot([Pb], [hi], pen=None, symbol="o", symbolSize=6, symbolBrush=GREEN, symbolPen=None, name="σ_P")
        if c["pmin"] < 3:
            for k in range(1, 4):
                pk = 1 / (k * SIDEREAL)
                if c["pmin"] < pk < c["pmax"]:
                    pi.addItem(pg.InfiniteLine(px(pk), angle=90, pen=pg.mkPen(BLUE, width=1, style=Qt.DotLine)))
        pi.addItem(pg.InfiniteLine(px(P), angle=90, pen=pg.mkPen(ORANGE, width=1.5)))
        pi.enableAutoRange()
        if norm == "Bayes log-post":
            has_win = any(ck.isChecked() and k in self.win for k, ck in (("win", self.ck_win), ("win_c", self.ck_winc)))
            pi.setYRange(lo - (hi - lo) if has_win else lo, 2, padding=0)

    def draw_fold(self):
        c = self.cache; r = c["r"]; pi = self.p_fd.getPlotItem()
        P, ph, xc, med = self.fold_data()
        pi.plot(np.r_[ph, ph + 1], np.r_[r, r], pen=None, symbol="o", symbolSize=2, symbolPen=None,
                symbolBrush=(150, 150, 150))
        pi.plot(np.r_[xc, xc + 1], np.r_[med, med], pen=pg.mkPen(RED, width=1.5), symbol="o", symbolSize=5,
                symbolBrush=RED, symbolPen=None, connect="finite")
        lo, hi = np.percentile(r, [1, 99])
        pi.setYRange(lo, hi); pi.setXRange(0, 2, padding=0.01)
        pi.setTitle(f"folded at P = {P:.6g} d", size="9pt")

    def draw_post(self):
        if not self.norm().startswith("Bayes"):
            self.p_post.hide(); return
        self.p_post.show()
        b = self.cache["bayes"]; pi = self.p_post.getPlotItem()
        pi.plot(1 / b["ff"], b["dens"] * b["ff"] ** 2, pen=pg.mkPen("#8e44ad", width=1.5))
        Pm = 1 / b["fmean"]
        pi.setXRange(Pm - 6 * b["sP"], Pm + 6 * b["sP"], padding=0)
        pi.enableAutoRange(axis="y")
        pi.setTitle(f"local p(P|D):  σ_P = {b['sP']:.3g} d  (white noise, σ marginalised)", size="9pt")

    def on_ls_click(self, ev):
        if ev.button() != Qt.LeftButton or self.cache is None or ev.double():
            return
        vb = self.p_ls.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = vb.mapSceneToView(ev.scenePos()).x()
        P = 10 ** x if self.ls_log else x
        if self.cache["pmin"] <= P <= self.cache["pmax"]:
            self.Pfold = float(P)
            self.redraw()

    # ------------------------------------------------------------------ summary and output
    def summary_text(self):
        c = self.cache; s = c["settings"]
        t, r, Pb = c["t"], c["r"], 1 / c["fbest"]
        nights = np.unique(night_id(t), return_counts=True)[1]
        P, _, _, med = self.fold_data()
        pp = 1e3 * (np.nanmax(med) - np.nanmin(med))
        an = self.core.analytic(c, self.co_sig.currentText())
        norm = self.norm(); b = c["bayes"]
        L = [f"{os.path.basename(self.name)}   MJD {s['sel'][0]:.0f}–{s['sel'][1]:.0f}"
             + (f"   excl. {s['exclude_txt']}" if s["exclude_txt"] else ""),
             f"N = {c['N']},  nights = {nights.size},  median pts/night = {np.median(nights):.0f}",
             f"T = {c['T']:.0f} d,  resolution P²/T = {Pb ** 2 / c['T']:.4g} d",
             f"detrend {s['detrend_txt']}, clip {'on' if s['clip'] else 'off'}"
             + (f", orbit {s['orbit']:g} d removed" if s["orbit"] else ""),
             f"K = {c['K']},  norm = {norm},  P range {c['pmin']:g}–{c['pmax']:g} d"
             + (f", searched P ≥ {c['pcut']:.3g} d" if c["pcut"] else ""),
             f"best P = {Pb:.6g} d   (f = {c['fbest']:.6f} 1/d)",
             f"power: std {c['pbest']:.4f}"
             + ("" if norm in ("standard", "Bayes post") else f",  shown {self.nrm(c['pbest']):.4g}"),
             f"robust rms = {1e3 * robust_sigma(r):.2f} mmag",
             f"fold P = {P:.6g} d,  binned p-p = {pp:.1f} mmag"]
        if an is not None:
            fs = "< 1e-16" if an[1] < 1e-16 else f"{an[1]:.2g}"
            L.append(f"FAP = {fs}  ({an[2]}; white noise only!)")
        L.append(f"Bayes posterior σ_P = {b['sP']:.3g} d  (white noise)")
        R = []
        if self.unc is not None:
            R.append(f"σ_P = {self.unc['sigma']:.3g} d  ({self.unc['method']}, n = {self.unc['n']})")
        if self.red is not None:
            d = self.red
            R += [f"red-noise null: α = {d['alpha']:.3g} ({d['how']}), {d['nsim']} sims",
                  f"  rms_red {1e3 * d['rms_red']:.2f} mmag, σ_white {1e3 * d['sw']:.2f} mmag",
                  f"  99% max power {d['lev99']:.4f} (std),  p(max ≥ peak) = {d['pval']:.2g}"]
        if self.boot is not None:
            R.append("σ_P comparison [d]:")
            R += [f"  {name:18s}{sg:9.4g}   (n = {n})" for name, sg, n in self.boot]
        row = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), file=os.path.basename(self.name),
                   mjd_start=round(s["sel"][0], 2), mjd_end=round(s["sel"][1], 2), exclude=s["exclude_txt"],
                   N=c["N"], nights=nights.size, pts_per_night=float(np.median(nights)), T_d=round(c["T"], 2),
                   detrend=s["detrend_txt"], clip5sig=s["clip"], orbit_sub_P=s["orbit"] if s["orbit"] else np.nan,
                   K=c["K"], normalisation=norm, Pmin=c["pmin"], Pmax=c["pmax"],
                   P_best=Pb, f_best=c["fbest"], power_std=c["pbest"],
                   power_norm=np.nan if norm == "Bayes post" else self.nrm(c["pbest"]),
                   FAP=an[1] if an else np.nan, FAP_method=an[2] if an else "none",
                   sigma_P=self.unc["sigma"] if self.unc else np.nan,
                   sigma_P_method=self.unc["method"] if self.unc else "", sigma_P_bayes=b["sP"],
                   P2_over_T=Pb ** 2 / c["T"], rms_mmag=1e3 * robust_sigma(r), P_fold=P, profile_pp_mmag=pp,
                   red_alpha=self.red["alpha"] if self.red else np.nan,
                   red_maxpow99=self.red["lev99"] if self.red else np.nan,
                   red_pvalue=self.red["pval"] if self.red else np.nan)
        return L, R, row

    def outdir_now(self):
        d = self.le_out.text().strip() or "explorer_output"
        os.makedirs(d, exist_ok=True)
        return d

    def stem(self):
        base = os.path.splitext(os.path.basename(self.name))[0]
        s = self.cache["settings"]
        return os.path.join(self.outdir_now(), f"{base}_{s['sel'][0]:.0f}-{s['sel'][1]:.0f}_K{self.cache['K']}")

    def save_panels(self):
        if self.cache is None:
            return
        stem = self.stem(); out = []
        panels = [("lightcurve", self.p_lc), ("detrended", self.p_dt), ("periodogram", self.p_ls),
                  ("folded", self.p_fd)]
        if self.p_post.isVisible():
            panels.append(("posterior", self.p_post))
        for key, w in panels:
            ex = pg.exporters.ImageExporter(w.getPlotItem())
            ex.parameters()["width"] = 1600
            ex.export(f"{stem}_{key}.png"); out.append(f"{stem}_{key}.png")
        self.grab().save(f"{stem}_full.png")
        L, R, _ = self.summary_text()
        with open(f"{stem}_summary.txt", "w") as fh:
            fh.write("\n".join(L + [""] + R) + "\n")
        self.set_status(f"saved {len(out)} panels + full window + summary: {stem}_*")

    def save_row(self):
        if self.cache is None:
            return
        _, _, row = self.summary_text()
        csv_path = os.path.join(self.outdir_now(), "explorer_results.csv")
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RESULT_COLS)
            if new:
                w.writeheader()
            w.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in row.items()})
        try:
            Table.read(csv_path, format="ascii.csv").write(csv_path[:-4] + ".ecsv", format="ascii.ecsv",
                                                           overwrite=True)
        except Exception as err:                                        # noqa: BLE001
            print("[save row] ECSV mirror not written:", err)
        self.set_status(f"row appended to {csv_path}")

    def closeEvent(self, ev):
        if self.job is not None:
            self.job.cancel = True
            self.job.wait(3000)
        super().closeEvent(ev)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", default=None, help="light curve (optional: otherwise pick from the list)")
    ap.add_argument("--dir", default=".", help="folder whose *.dat files are listed")
    ap.add_argument("--pmin", type=float, default=10.0)
    ap.add_argument("--pmax", type=float, default=60.0)
    ap.add_argument("--porb", type=float, default=None, help="orbital period for 'subtract orbit'")
    ap.add_argument("--exclude", default="", help='MJD intervals to drop, e.g. "56550-56800, 60300-60470"')
    ap.add_argument("--alias", default="", help="alias cut: periods below are shaded and not searched ('auto' = 2 d)")
    ap.add_argument("--live", action="store_true", help="start with live update on")
    ap.add_argument("--outdir", default="explorer_output")
    a = ap.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = ExplorerQt(a.file, a.pmin, a.pmax, a.porb, workdir=a.dir, outdir=a.outdir, live=a.live,
                     exclude=a.exclude, alias=a.alias)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
