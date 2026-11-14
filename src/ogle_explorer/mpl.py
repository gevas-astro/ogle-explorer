"""Interactive explorer for OGLE light curves (instructor reference implementation, v2).

    ogle-explorer-mpl                          # pick a *.dat file from the list in the window
    ogle-explorer-mpl rxj0520.dat --exclude "60300-60470"
    ogle-explorer-mpl smc_x3.dat --pmin 0.1 --pmax 3 --porb 44.9   # NRP range

Layout
  Top     full light curve: drag to select the interval; grey bands = excluded intervals.
  Middle  detrended selection (left), periodogram (right; click on it to choose the fold period).
  Bottom  folded light curve with binned medians (left), summary numbers (right).

Controls
  files / path      list of *.dat in the working folder (click to load) and a path box (Enter to load).
  live update       OFF by default: parameter changes are only applied with the [update] button.
  detrend           running median of half-width "detrend ±[d]" (unticked = only the median is removed).
  harmonics K       LS nterms.  Both can be dragged or typed in the box next to the slider (Enter).  Pmin / Pmax [d] search range, Porb [d] for "subtract orbit".
  exclude           MJD intervals left out, e.g. "56550-56800, 60300-60470".
  alias P<          periods below this [d] are the alias region: shaded grey, not used for the peak, the FAP
                    trials or the red-noise maxima. Empty = off, 'auto' = 2 sidereal days, or a number.
  red α             slope of the red-noise PSD f^-α for the null: a number (e.g. 1, 2) or 'fit' / empty
                    = PSRESP-lite fit of α over 0.5 … 3.5 to the residual periodogram.
  window, window @peak   spectral window W(f) itself ("win") and centred on the peak, max[W(f - f_peak), W(f + f_peak)] ("win_c").
  normalisation     standard, psd, model, log (all exact transforms of the standard power P),
                    Bayes log-post  ln p(f|D) - max,  with p(f|D) ∝ (1 - P)^{-(N-2K-1)/2} (σ marginalised),
                    Bayes post      normalised posterior density over the frequency grid.
  significance      analytic, WHITE NOISE ONLY: Baluev (astropy, K = 1) or the Beta single-frequency
                    tail  P ~ Beta(K, (N-2K-1)/2)  with M ≈ T·Δf independent trials.  99 % and 99.9 % lines.
  red-noise null    TK95 red noise at the real times (slope "red α": a number, or "fit" = PSRESP-lite),
                    detrended like the data, rms from the residuals after removing the best model,
                    plus white noise; "N sims" realisations -> 95 / 99 % of the maximum power.
  season jackknife, season boot   σ_P from leaving out / resampling whole observing seasons.
  boot compare      point vs night vs season bootstrap vs season jackknife vs analytic white-noise σ_P.
  save panels       each panel as its own PNG (+ the full window and a text summary) in --outdir.
  save row          append one row per analysis to explorer_results.csv (and .ecsv) in --outdir.

numpy, matplotlib, astropy (LombScargle, Table) and scipy.stats (Beta tail; K = 1 works without it)
plus the package modules tools.py and core.py.
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector, Slider, TextBox, CheckButtons, Button, RadioButtons
from astropy.timeseries import LombScargle
from astropy.table import Table

from . import tools as ot
from .core import (load, running_median, robust_sigma, folded_template, night_id, season_id,
                   parse_exclude, to_norm, beta_sf, beta_isf, SIDEREAL, YEAR, NORMS, SIGS, FAPS, RESULT_COLS)

CB_LABELS = ["live update", "detrend", "subtract orbit", "clip 5σ", "window", "window @peak"]
LIVE, DETREND, ORBIT, CLIP, WINDOW, WINDOW_C = range(6)


class FastTextBox(TextBox):
    """matplotlib's TextBox re-renders the WHOLE figure (canvas.draw) on every keystroke, which is slow with
    ~10^4-10^5 plotted points. Here the cursor/text update is drawn into the box's own axes and blitted."""

    def _rendercursor(self):
        fig = self.ax.figure
        canvas = fig.canvas
        if not getattr(canvas, "supports_blit", False) or fig._get_renderer() is None:
            return super()._rendercursor()
        full_draw = canvas.draw
        canvas.draw = lambda *a, **k: None          # suppress the full redraw inside the parent method
        try:
            super()._rendercursor()
        finally:
            canvas.draw = full_draw
        fig.draw_artist(self.ax)                     # repaint only this box ...
        canvas.blit(self.ax.bbox)                    # ... and push it to the screen


class FastSlider(Slider):
    """matplotlib's Slider also redraws the WHOLE figure on every drag step; repaint only its own axes.
    The numeric read-out is shown in a separate text box (the built-in label sits outside the axes)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.valtext.set_visible(False)

    def set_val(self, val):
        fig = self.ax.figure; canvas = fig.canvas
        if not getattr(canvas, "supports_blit", False) or fig._get_renderer() is None:
            return super().set_val(val)
        self.drawon = False
        try:
            super().set_val(val)
        finally:
            self.drawon = True
        fig.draw_artist(self.ax)
        canvas.blit(self.ax.bbox)


# ------------------------------------------------------------------ the tool
class Explorer:
    MAXF = 40000                        # cap on the number of trial frequencies (keeps K > 1 responsive)

    def __init__(self, path=None, pmin=10.0, pmax=60.0, porb=None, workdir=".", outdir="explorer_output",
                 live=False, exclude="", alias=""):
        self.workdir, self.outdir = workdir, outdir
        self.t = self.y = self.e = np.array([])
        self.name = ""
        self.sel = (0.0, 1.0)
        self.cache = None
        self.Pfold = None
        self.unc = None                 # dict(method, sigma, n)
        self.boot = None                # list of (method, sigma, n) from "boot compare"
        self.red = None                 # red-noise null result
        self.window = self.window_c = None
        self.stale = False
        self._loading = False
        self.span_patch = None
        self.exc_patches = []

        self.fig = plt.figure(figsize=(15, 10))
        mgr = getattr(self.fig.canvas, "manager", None)
        self._mgr = mgr
        g = self.fig.add_gridspec(3, 2, left=0.05, right=0.98, top=0.95, bottom=0.375, hspace=0.55, wspace=0.15)
        self.ax_lc = self.fig.add_subplot(g[0, :])
        self.ax_dt = self.fig.add_subplot(g[1, 0])
        self.ax_ls = self.fig.add_subplot(g[1, 1])
        self.ax_fd = self.fig.add_subplot(g[2, 0])
        self.ax_tx = self.fig.add_subplot(g[2, 1]); self.ax_tx.axis("off")
        self.lc_line, = self.ax_lc.plot([], [], ".", ms=1.5, color="k")
        self.ax_lc.set_xlabel("MJD"); self.ax_lc.set_ylabel("I [mag]")
        self.span = SpanSelector(self.ax_lc, self.on_select, "horizontal", useblit=True,
                                 props=dict(alpha=0.3, facecolor="C0"), interactive=True, drag_from_anywhere=True)
        self.status = self.fig.text(0.98, 0.977, "", ha="right", va="center", fontsize=9, color="C3")
        self._build_controls(pmin, pmax, porb, live, exclude, alias)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        if path is None and self.files:
            path = self.files[0]
        if path:
            self.load_file(path)
        else:
            self.set_status("no *.dat file here: type a path in the 'path' box")

    # ---------------------------------------------------------------- widgets
    def _build_controls(self, pmin, pmax, porb, live, exclude, alias=""):
        F = self.fig
        A = F.add_axes
        # files (edit 7)
        self.files = sorted(glob.glob(os.path.join(self.workdir, "*.dat")))[:10]
        labels = [os.path.basename(p) for p in self.files] or ["(no *.dat here)"]
        wd = os.path.abspath(self.workdir)
        F.text(0.02, 0.31, "files in …" + wd[-28:] if len(wd) > 30 else f"files in {wd}", fontsize=8)
        self.rb_file = RadioButtons(A([0.02, 0.09, 0.16, 0.21]), labels, active=0)
        self.rb_file.on_clicked(self.on_file)
        self.tb_path = FastTextBox(A([0.05, 0.035, 0.13, 0.03]), "path ", initial="")
        self.tb_path.on_submit(lambda s: self.load_file(s.strip()) if s.strip() else None)

        # analysis parameters
        self.s_det = FastSlider(A([0.28, 0.27, 0.145, 0.022]), "detrend ±[d]", 1, 100, valinit=50, valstep=0.5)
        self.s_nt = FastSlider(A([0.28, 0.225, 0.145, 0.022]), "harmonics K", 1, 5, valinit=1, valstep=1)
        self.tb_det = FastTextBox(A([0.43, 0.265, 0.03, 0.03]), "", initial="50")
        self.tb_nt = FastTextBox(A([0.43, 0.22, 0.03, 0.03]), "", initial="1")
        tb = lambda r, lab, init: FastTextBox(A(r), lab, initial=init)
        self.tb_min = tb([0.28, 0.165, 0.06, 0.03], "Pmin [d] ", f"{pmin:g}")
        self.tb_max = tb([0.40, 0.165, 0.06, 0.03], "Pmax [d] ", f"{pmax:g}")
        self.tb_orb = tb([0.28, 0.12, 0.06, 0.03], "Porb [d] ", "" if porb is None else f"{porb:g}")
        self.tb_alpha = tb([0.40, 0.12, 0.06, 0.03], "red α ", "fit")
        self.tb_nsim = tb([0.28, 0.075, 0.06, 0.03], "N sims ", "50")
        self.tb_nboot = tb([0.40, 0.075, 0.06, 0.03], "N boot ", "200")
        self.tb_exc = tb([0.28, 0.03, 0.10, 0.03], "exclude ", exclude)
        self.tb_alias = tb([0.43, 0.03, 0.03, 0.03], "alias P< ", alias)

        # check boxes (edits 1, 2)
        F.text(0.50, 0.31, "options", fontsize=8)
        self.cb = CheckButtons(A([0.50, 0.09, 0.12, 0.21]), CB_LABELS, [live, True, False, True, False, False])
        # normalisation (edit 3) and analytic significance (edit 4)
        F.text(0.63, 0.31, "normalisation", fontsize=8)
        self.rb_norm = RadioButtons(A([0.63, 0.09, 0.11, 0.21]), NORMS, active=0)
        F.text(0.75, 0.31, "analytic sig. (white)", fontsize=8)
        self.rb_sig = RadioButtons(A([0.75, 0.17, 0.10, 0.13]), SIGS, active=1)
        for w in (self.rb_file, self.cb, self.rb_norm, self.rb_sig):
            for lab in w.labels:
                lab.set_fontsize(8)

        # buttons (edits 1, 5, 6)
        B = lambda r, lab: Button(A(r), lab)
        self.b_red = B([0.75, 0.115, 0.10, 0.04], "red-noise null")
        self.b_cmp = B([0.75, 0.065, 0.10, 0.04], "boot compare")
        self.b_go = B([0.87, 0.245, 0.11, 0.05], "update")
        self.b_go.label.set_fontweight("bold")
        self.b_jk = B([0.87, 0.19, 0.11, 0.04], "season jackknife")
        self.b_sb = B([0.87, 0.145, 0.11, 0.04], "season boot")
        self.b_sp = B([0.87, 0.075, 0.11, 0.04], "save panels")
        self.b_row = B([0.87, 0.03, 0.11, 0.04], "save row")

        # callbacks
        self.s_det.on_changed(lambda v: self._slider_moved(self.s_det, self.tb_det, v))
        self.s_nt.on_changed(lambda v: self._slider_moved(self.s_nt, self.tb_nt, v))
        self.tb_det.on_submit(lambda txt: self._box_entered(self.s_det, self.tb_det, txt))
        self.tb_nt.on_submit(lambda txt: self._box_entered(self.s_nt, self.tb_nt, txt))
        for w in (self.tb_min, self.tb_max, self.tb_orb, self.tb_exc, self.tb_alias):
            w.on_submit(lambda s: self.changed())
        self.cb.on_clicked(self.on_check)
        self.rb_norm.on_clicked(lambda lab: self.redraw())
        self.rb_sig.on_clicked(lambda lab: self.redraw())
        self.b_go.on_clicked(lambda ev: self.update())
        self.b_jk.on_clicked(lambda ev: self.run_uncertainty("jackknife"))
        self.b_sb.on_clicked(lambda ev: self.run_uncertainty("season"))
        self.b_cmp.on_clicked(lambda ev: self.boot_compare())
        self.b_red.on_clicked(lambda ev: self.red_noise_null())
        self.b_sp.on_clicked(lambda ev: self.save_panels())
        self.b_row.on_clicked(lambda ev: self.save_row())

    def _slider_moved(self, slider, box, v):
        """Slider -> box (without firing the box's submit), then flag / recompute."""
        txt = f"{v:g}"
        if box.text != txt:
            box.eventson = False
            box.set_val(txt)
            box.eventson = True
        self.changed()

    def _box_entered(self, slider, box, txt):
        """Typed value (Enter) -> clipped to the slider range and step -> slider (which calls changed())."""
        try:
            v = float(txt)
        except ValueError:
            box.eventson = False; box.set_val(f"{slider.val:g}"); box.eventson = True
            self.set_status("not a number"); return
        v = min(max(v, slider.valmin), slider.valmax)
        v = round(v / slider.valstep) * slider.valstep
        if v == slider.val:                                   # nothing changes; just tidy the box
            box.eventson = False; box.set_val(f"{v:g}"); box.eventson = True
        else:
            slider.set_val(v)

    def opt(self, i):
        return self.cb.get_status()[i]

    def set_status(self, msg, flush=False):
        self.status.set_text(msg)
        if flush:
            self.fig.canvas.draw()
            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass
        else:
            self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- callbacks
    def on_file(self, label):
        if self._loading:
            return
        for p in self.files:
            if os.path.basename(p) == label:
                self.load_file(p)

    def on_check(self, label):
        if label == "live update":
            if self.opt(LIVE) and self.stale:
                self.update()
        elif label in ("window", "window @peak"):
            self.redraw()                         # display only, no recomputation
        else:
            self.changed()

    def changed(self):
        """A parameter that requires a recomputation changed."""
        if self.t.size == 0:
            return
        if self.opt(LIVE):
            self.update()
        elif not self.stale:                       # flag once; no full redraw per slider step / keystroke
            self.stale = True
            self.set_status("settings changed: press [update]")

    def on_select(self, x0, x1):
        if x1 - x0 < 5:                              # a click, not a drag: keep the old selection
            return
        self.sel = (x0, x1)
        self._draw_sel_patch()
        self.Pfold = None
        self.changed()

    def on_click(self, ev):
        if ev.inaxes is self.ax_ls and ev.xdata is not None and ev.button == 1 and self.cache is not None:
            self.Pfold = ev.xdata
            self.redraw()

    # ---------------------------------------------------------------- data
    def load_file(self, path):
        if not os.path.isfile(path):
            self.set_status(f"file not found: {path}")
            return
        try:
            t, y, e = load(path)
        except Exception as err:                    # noqa: BLE001
            self.set_status(f"cannot read {os.path.basename(path)}: {err}")
            return
        o = np.argsort(t)
        self.t, self.y, self.e = t[o], y[o], e[o]
        self.name = path
        self.sel = (self.t.min(), self.t.max())
        self.Pfold = None
        self.cache = None
        self.lc_line.set_data(self.t, self.y)
        pad = 0.01 * np.ptp(self.t)
        self.ax_lc.set_xlim(self.t.min() - pad, self.t.max() + pad)
        dy = 0.05 * np.ptp(self.y)
        self.ax_lc.set_ylim(self.y.max() + dy, self.y.min() - dy)
        self.ax_lc.set_title(f"{os.path.basename(path)}: drag to select the interval", fontsize=10)
        self._draw_sel_patch()
        if self._mgr is not None:
            try:
                self._mgr.set_window_title(f"OGLE explorer - {os.path.basename(path)}")
            except Exception:
                pass
        # keep the file list in sync (without re-triggering a load)
        for i, p in enumerate(self.files):
            if os.path.abspath(p) == os.path.abspath(path) and self.rb_file.value_selected != os.path.basename(p):
                self._loading = True
                self.rb_file.set_active(i)
                self._loading = False
        self.update()

    def _draw_sel_patch(self):
        if self.span_patch is not None:
            self.span_patch.remove()
        self.span_patch = self.ax_lc.axvspan(*self.sel, color="C1", alpha=0.15)

    def exclusions(self):
        try:
            return parse_exclude(self.tb_exc.text)
        except ValueError:
            self.set_status("bad 'exclude' (use e.g. 56550-56800, 60300-60470)")
            return []

    def alias_cut(self):
        """Periods below this are treated as the alias region: shaded grey and excluded from the peak search.
        '' / 'off' = none, 'auto' = 2 sidereal days (pseudo-Nyquist of nightly sampling), or a number [d]."""
        s = self.tb_alias.text.strip().lower()
        if s in ("", "off", "none", "0"):
            return None
        if s == "auto":
            return 2 / SIDEREAL
        try:
            return float(s)
        except ValueError:
            self.set_status("alias P< must be a number, 'auto' or empty")
            return None

    def settings(self):
        try:
            porb = float(self.tb_orb.text)
        except ValueError:
            porb = None
        return dict(detrend=f"median ±{self.s_det.val:g} d" if self.opt(DETREND) else "none",
                    clip=self.opt(CLIP), orbit=porb if (self.opt(ORBIT) and porb) else None,
                    exclude=self.tb_exc.text.strip())

    def prepare(self):
        m = (self.t >= self.sel[0]) & (self.t <= self.sel[1])
        for a, b in self.exclusions():
            m &= ~((self.t > a) & (self.t < b))
        t, y, e = self.t[m], self.y[m], self.e[m]
        if t.size < 20:
            return t, y, e
        r = y - running_median(t, y, self.s_det.val) if self.opt(DETREND) else y - np.median(y)
        s = self.settings()
        if s["orbit"]:
            r = r - folded_template(t, r, s["orbit"])
        if s["clip"]:
            k = np.abs(r - np.median(r)) < 5 * robust_sigma(r)
            t, r, e = t[k], r[k], e[k]
        return t, r, e

    # ---------------------------------------------------------------- analysis
    def update(self, recompute=True):
        if self.t.size == 0:
            return
        if recompute or self.cache is None:
            self.set_status("computing …", flush=True)
            if not self.compute():
                self.redraw()
                return
        self.set_status("")
        self.redraw()

    def compute(self):
        try:
            pmin, pmax = float(self.tb_min.text), float(self.tb_max.text)
        except ValueError:
            self.set_status("bad Pmin / Pmax"); return False
        if not 0 < pmin < pmax:
            self.set_status("need 0 < Pmin < Pmax"); return False
        t, r, e = self.prepare()
        if t.size < 20:
            self.cache = None
            self.set_status(f"only {t.size} points in the selection"); return False
        T = np.ptp(t); K = int(self.s_nt.val); N = t.size
        nf = int(min(self.MAXF, 5 * T * (1 / pmin - 1 / pmax)))
        f = np.linspace(1 / pmax, 1 / pmin, max(nf, 500))
        ls = LombScargle(t, r, nterms=K)               # standard normalisation; others are exact transforms
        p = ls.power(f)
        pcut = self.alias_cut()
        search = np.ones(f.size, bool) if pcut is None else (1 / f >= pcut)
        if search.sum() < 10:
            self.set_status(f"alias cut {pcut:g} d leaves no periods to search"); return False
        ib = np.flatnonzero(search)[p[search].argmax()]; df = f[1] - f[0]
        ff = np.linspace(f[ib] - df, min(f[ib] + df, 1 / pcut if pcut else np.inf), 101)  # refine below the grid step
        pf = ls.power(ff)
        self.cache = dict(t=t, r=r, e=e, f=f, p=p, K=K, N=N, T=T, ls=ls, pmin=pmin, pmax=pmax,
                          pcut=pcut, search=search, fs=f[search],
                          fbest=ff[pf.argmax()], pbest=float(pf.max()),
                          chi2ref=float(np.sum((r - r.mean()) ** 2)), settings=self.settings())
        if self.Pfold is not None and not pmin <= self.Pfold <= pmax:
            self.Pfold = None
        self.unc = self.boot = self.red = self.window = self.window_c = None    # previous analysis
        self.stale = False
        return True

    # significance ---------------------------------------------------
    def analytic(self):
        """Return (levels in standard power for FAPS, FAP of the peak, label) or None."""
        c = self.cache; meth = self.rb_sig.value_selected
        if meth == "none" or c is None:
            return None
        f, K, N, T = c["fs"], c["K"], c["N"], c["T"]           # searched range only
        if meth == "Baluev" and K == 1:
            lev = np.atleast_1d(c["ls"].false_alarm_level(list(FAPS), minimum_frequency=f[0],
                                                          maximum_frequency=f[-1], method="baluev"))
            fap = float(c["ls"].false_alarm_probability(c["pbest"], minimum_frequency=f[0],
                                                        maximum_frequency=f[-1], method="baluev"))
            return lev, fap, "Baluev"
        M = max(T * (f[-1] - f[0]), 1.0)
        q = -np.expm1(np.log1p(-np.array(FAPS)) / M)        # single-trial tail for the requested FAP
        lev = beta_isf(q, K, N)
        tail = beta_sf(c["pbest"], K, N)
        fap = float(-np.expm1(M * np.log1p(-tail))) if np.isfinite(tail) else np.nan
        lab = f"Beta(K,(N-2K-1)/2), M≈{M:.0f}" + (" [Baluev needs K=1]" if meth == "Baluev" else "")
        return lev, fap, lab

    def red_noise_null(self):
        c = self.cache
        if c is None:
            return
        try:
            nsim = max(int(self.tb_nsim.text), 5)
        except ValueError:
            nsim = 50
        t, r, f, K = c["t"], c["r"], c["fs"], c["K"]
        resid = r - c["ls"].model(t, c["fbest"])           # rms "from residuals": best model removed
        sw = ot.white_level(t, resid)
        if sw is None:                                      # sparse sampling: use the photometric errors
            sw = float(np.median(c["e"]))
        sw = min(sw, 0.95 * resid.std())
        rms_red = float(np.sqrt(max(resid.var() - sw ** 2, 1e-12)))
        txt = self.tb_alpha.text.strip().lower()
        if txt in ("", "fit"):
            self.set_status("fitting the red-noise slope (PSRESP-lite) …", flush=True)
            alpha = ot.fit_red_noise(t, resid, sw, alphas=np.arange(0.5, 3.51, 0.5), nsim=10,
                                     rng=np.random.default_rng(3))[0]
            how = "fitted"
        else:
            try:
                alpha = float(txt); how = "set"
            except ValueError:
                self.set_status("red α must be a number or 'fit'"); return
        rng = np.random.default_rng(1)
        det = self.opt(DETREND); h = self.s_det.val
        maxp = np.empty(nsim)
        t0 = time.time()
        for i in range(nsim):
            if i % 5 == 0:
                self.set_status(f"red-noise sims {i}/{nsim} (α = {alpha:g}) …", flush=True)
            x = ot.tk95_sampled(t, alpha, 1.0, rng)
            if det:                                         # same detrending as the data
                x = x - running_median(t, x, h)
            x = (x - x.mean()) / x.std() * rms_red + rng.normal(0, sw, t.size)
            maxp[i] = LombScargle(t, x, nterms=K).power(f).max()
        self.red = dict(alpha=alpha, how=how, rms_red=rms_red, sw=sw, nsim=nsim, maxp=maxp,
                        lev95=float(np.percentile(maxp, 95)), lev99=float(np.percentile(maxp, 99)),
                        pval=float((np.sum(maxp >= c["pbest"]) + 1) / (nsim + 1)))
        print(f"[red-noise null] alpha={alpha:g} ({how}), rms_red={1e3*rms_red:.2f} mmag, "
              f"sigma_white={1e3*sw:.2f} mmag, {nsim} sims in {time.time()-t0:.1f} s: "
              f"max-power 95%={self.red['lev95']:.4f} 99%={self.red['lev99']:.4f}, "
              f"peak {c['pbest']:.4f} -> p = {self.red['pval']:.3g}")
        self.set_status("")
        self.redraw()

    # uncertainty ----------------------------------------------------
    def _peak(self, t, r, f0):
        c = self.cache; K, T = c["K"], c["T"]
        ls = LombScargle(t, r, nterms=K)
        ff = np.linspace(f0 - 3 / T, f0 + 3 / T, 241); ff = ff[ff > 0]
        i = ls.power(ff).argmax(); d = ff[1] - ff[0]
        f2 = np.linspace(ff[i] - d, ff[i] + d, 41)
        return 1 / f2[ls.power(f2).argmax()]

    def sigma_P(self, kind, nboot):
        c = self.cache; t, r, f0 = c["t"], c["r"], c["fbest"]
        rng = np.random.default_rng(42)
        if kind == "point":
            Ps = [self._peak(t[k], r[k], f0) for k in (rng.integers(0, t.size, t.size) for _ in range(nboot))]
            return float(np.std(Ps, ddof=1)), t.size
        g = night_id(t) if kind == "night" else season_id(t)
        ug, inv = np.unique(g, return_inverse=True)
        blocks = [np.flatnonzero(inv == k) for k in range(ug.size)]
        n = ug.size
        if n < 3:
            return np.nan, n
        if kind == "jackknife":
            Ps = np.array([self._peak(t[inv != k], r[inv != k], f0) for k in range(n)])
            return float(np.sqrt((n - 1) / n * np.sum((Ps - Ps.mean()) ** 2))), n
        Ps = []
        for _ in range(nboot):
            k = np.concatenate([blocks[j] for j in rng.integers(0, n, n)])
            Ps.append(self._peak(t[k], r[k], f0))
        return float(np.std(Ps, ddof=1)), n

    def sigma_analytic(self):
        """Montgomery & O'Donoghue (1999): sigma_f = sqrt(6/N) sigma / (pi T A); white noise, K = 1 sine."""
        c = self.cache; t, r, fb = c["t"], c["r"], c["fbest"]
        chi2, A = ot.sine_chi2(t, r, fb)
        sig = np.sqrt(chi2 / (t.size - 3))
        return float(np.sqrt(6 / t.size) * sig / (np.pi * c["T"] * A) / fb ** 2)

    def nboot(self):
        try:
            return max(int(self.tb_nboot.text), 10)
        except ValueError:
            return 200

    def run_uncertainty(self, kind):
        if self.cache is None:
            return
        name = {"jackknife": "season jackknife", "season": "season boot"}[kind]
        self.set_status(f"{name} …", flush=True)
        s, n = self.sigma_P(kind, self.nboot())
        if not np.isfinite(s):
            self.set_status(f"{name}: need ≥ 3 seasons in the selection (have {n})"); return
        self.unc = dict(method=name, sigma=s, n=n)
        print(f"[{name}] sigma_P = {s:.4g} d  ({n} seasons)")
        self.set_status("")
        self.redraw()

    def boot_compare(self):
        if self.cache is None:
            return
        nb = self.nboot(); rows = []
        for kind, name in [("point", "point boot"), ("night", "night boot"), ("season", "season boot"),
                           ("jackknife", "season jackknife")]:
            self.set_status(f"boot compare: {name} …", flush=True)
            s, n = self.sigma_P(kind, nb)
            rows.append((name, s, n))
        rows.append(("analytic white", self.sigma_analytic(), self.cache["N"]))
        self.boot = rows
        print("[boot compare] P = %.5f d, %d resamples" % (1 / self.cache["fbest"], nb))
        for name, s, n in rows:
            print(f"   {name:18s} sigma_P = {s:.4g} d   (n = {n})")
        if np.isfinite(rows[2][1]):
            self.unc = dict(method="season boot", sigma=rows[2][1], n=rows[2][2])
        self.set_status("")
        self.redraw()

    # ---------------------------------------------------------------- drawing
    def display(self):
        """Power on the grid in the selected normalisation, and the y label."""
        c = self.cache; norm = self.rb_norm.value_selected
        if norm == "Bayes post":
            L = to_norm(c["p"], "Bayes log-post", c["N"], c["K"], c["chi2ref"], c["pbest"])
            w = np.exp(L)
            return w / (w.sum() * (c["f"][1] - c["f"][0])), "p(f | D)  [d]"
        lab = {"standard": "LS power", "psd": "LS power (psd)", "model": "LS power (model)",
               "log": "LS power (log)", "Bayes log-post": "ln p(f|D) − max"}[norm]
        return to_norm(c["p"], norm, c["N"], c["K"], c["chi2ref"], c["pbest"]), f"{lab}, K={c['K']}"

    def bayes_local(self):
        """Posterior p(f|D) on a dense grid of ±0.5/T around the peak: the global grid (5x oversampled)
        is much coarser than the posterior width when N is large. Returns dict with sigma_P (Bayes)."""
        c = self.cache
        if c.get("bayes") is None:
            T = c["T"]
            ff = np.linspace(c["fbest"] - 0.5 / T, c["fbest"] + 0.5 / T, 4001)
            pl = c["ls"].power(ff)
            L = to_norm(pl, "Bayes log-post", c["N"], c["K"], c["chi2ref"], pl.max())
            w = np.exp(L); w /= w.sum()
            fm = float((w * ff).sum()); sf = float(np.sqrt((w * (ff - fm) ** 2).sum()))
            c["bayes"] = dict(ff=ff, dens=w / (ff[1] - ff[0]), fmean=fm, sf=sf, sP=sf / fm ** 2)
        return c["bayes"]

    def nrm(self, p):
        c = self.cache
        return float(to_norm(p, self.rb_norm.value_selected, c["N"], c["K"], c["chi2ref"], c["pbest"]))

    def fold_data(self):
        c = self.cache; t, r = c["t"], c["r"]
        P = self.Pfold or 1 / c["fbest"]
        ph = ((t - t.min()) / P) % 1
        nb = 20; b = np.minimum((ph * nb).astype(int), nb - 1)
        med = np.array([np.median(r[b == k]) if (b == k).any() else np.nan for k in range(nb)])
        return P, ph, (np.arange(nb) + 0.5) / nb, med

    def draw_lc(self, a):
        """Stand-alone light-curve panel (for saving)."""
        a.plot(self.t, self.y, ".", ms=1.5, color="k")
        a.axvspan(*self.sel, color="C1", alpha=0.15)
        for x0, x1 in self.exclusions():
            a.axvspan(x0, x1, color="0.5", alpha=0.35)
        dy = 0.05 * np.ptp(self.y)
        a.set_ylim(self.y.max() + dy, self.y.min() - dy)
        a.set_xlabel("MJD"); a.set_ylabel("I [mag]"); a.set_title(os.path.basename(self.name), fontsize=10)

    def draw_dt(self, a):
        c = self.cache; s = c["settings"]
        a.plot(c["t"], c["r"], ".", ms=2, color="k"); a.invert_yaxis()
        extra = (", orbit removed" if s["orbit"] else "") + (", 5σ clip" if s["clip"] else "")
        a.set_title(f"detrend: {s['detrend']}{extra};  N = {c['N']}", fontsize=9)
        a.set_xlabel("MJD"); a.set_ylabel("Δ I [mag]")

    def draw_ls(self, a):
        c = self.cache; f = c["f"]; norm = self.rb_norm.value_selected
        y, ylab = self.display()
        a.plot(1 / f, y, lw=0.7, color="k")
        Pb = 1 / c["fbest"]; P = self.Pfold or Pb
        if norm != "Bayes post":
            an = self.analytic()
            if an is not None and np.all(np.isfinite(an[0])):
                for lev, fap, ls_ in zip(an[0], FAPS, ("--", ":")):
                    a.axhline(self.nrm(lev), color="C0", ls=ls_, lw=0.9, label=f"FAP_W {100*(1-fap):g}")
            if self.red is not None:
                a.axhline(self.nrm(self.red["lev99"]), color="C3", lw=1.0, label="FAP_R 99")
        if self.unc is not None:
            s = self.unc["sigma"]
            a.axvspan(Pb - s, Pb + s, color="C2", alpha=0.35, lw=0)
            a.errorbar(Pb, np.nanmax(y), xerr=s, fmt="o", color="C2", ms=3, capsize=3,
                       label="σ_P")
        lo, hi = np.nanmin(y), np.nanmax(y)
        if self.opt(WINDOW):            # spectral window W(f) at the trial frequency itself
            if self.window is None:
                self.window = self._window(f)
            a.plot(1 / f, lo - self.window * (hi - lo), lw=0.6, color="m", label="win")
        if self.opt(WINDOW_C):          # alias pattern of a real signal at f_peak: aliases at |k f_s ± f_peak|
            if self.window_c is None:
                self.window_c = np.maximum(self._window(f - c["fbest"]), self._window(f + c["fbest"]))
            a.plot(1 / f, lo - self.window_c * (hi - lo), lw=0.6, color="r", label="win_c")
        if c["pcut"] is not None and c["pcut"] > c["pmin"]:
            a.axvspan(c["pmin"], min(c["pcut"], c["pmax"]), color="0.5", alpha=0.25, lw=0, label="alias")
        a.axvline(P, color="C1", lw=1)
        if norm.startswith("Bayes"):
            b = self.bayes_local()
            ins = a.inset_axes([0.58, 0.24, 0.40, 0.38])      # raised: its tick labels clear the main x axis
            Pl = 1 / b["ff"]; Pm = 1 / b["fmean"]
            ins.plot(Pl, b["dens"] * b["ff"] ** 2, color="C4", lw=1)        # density per unit P
            ins.set_xlim(Pm - 6 * b["sP"], Pm + 6 * b["sP"])
            ins.set_yticks([]); ins.tick_params(labelsize=6, pad=1)
            ins.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            ins.set_xlabel("P [d]", fontsize=6, labelpad=1)
            ins.text(0.02, 0.95, f"local p(P|D)\nσ_P = {b['sP']:.3g} d", transform=ins.transAxes,
                     fontsize=6.5, va="top", ha="left")
        if c["pmax"] / c["pmin"] > 20:
            a.set_xscale("log")
        if c["pmin"] < 3:
            for k in range(1, 4):
                if c["pmin"] < 1 / (k * SIDEREAL) < c["pmax"]:
                    a.axvline(1 / (k * SIDEREAL), color="b", ls=":", lw=0.7)
        if norm == "Bayes log-post":
            a.set_ylim(max(np.nanmin(y), -60), 2)
        a.set_xlabel("P [d]"); a.set_ylabel(ylab, fontsize=8)
        a.set_title("click to choose the fold period", fontsize=9)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=6, loc="upper left" if norm.startswith("Bayes") else "best", framealpha=0.8, handlelength=1.5, borderpad=0.3,
                     labelspacing=0.2, ncol=2 if len(a.get_legend_handles_labels()[0]) > 3 else 1)

    def _window(self, df):
        w = np.empty(df.size)
        for i in range(0, df.size, 2000):
            w[i:i + 2000] = ot.spectral_window(self.cache["t"], df[i:i + 2000])
        return w

    def draw_fold(self, a):
        c = self.cache; r = c["r"]
        P, ph, xc, med = self.fold_data()
        a.plot(np.r_[ph, ph + 1], np.r_[r, r], ".", ms=1.5, color="0.6")
        a.plot(np.r_[xc, xc + 1], np.r_[med, med], "o-", color="C3", ms=4)
        lo, hi = np.percentile(r, [1, 99]); a.set_ylim(hi, lo)
        a.set_xlabel("phase"); a.set_ylabel("Δ I [mag]"); a.set_title(f"folded at P = {P:.6g} d", fontsize=9)

    def summary(self):
        """Left and right text blocks plus the numbers for the results table."""
        c = self.cache; s = c["settings"]
        t, r, Pb = c["t"], c["r"], 1 / c["fbest"]
        nights = np.unique(night_id(t), return_counts=True)[1]
        P, _, _, med = self.fold_data()
        pp = 1e3 * (np.nanmax(med) - np.nanmin(med))
        an = self.analytic()
        L = [f"{os.path.basename(self.name)}   MJD {self.sel[0]:.0f}–{self.sel[1]:.0f}"
             + (f"  excl. {s['exclude']}" if s["exclude"] else ""),
             f"N = {c['N']},  nights = {nights.size},  median pts/night = {np.median(nights):.0f}",
             f"T = {c['T']:.0f} d,  resolution P²/T = {Pb**2 / c['T']:.4g} d",
             f"detrend {s['detrend']}, clip {'on' if s['clip'] else 'off'}"
             + (f", orbit {s['orbit']:g} d removed" if s["orbit"] else ""),
             f"K = {c['K']},  norm = {self.rb_norm.value_selected}",
             f"best P = {Pb:.6g} d  (f = {c['fbest']:.6f} 1/d)"
             + (f",  searched P ≥ {c['pcut']:.3g} d" if c["pcut"] else ""),
             f"power: std {c['pbest']:.4f}"
             + ("" if self.rb_norm.value_selected in ("standard", "Bayes post") else f",  shown {self.nrm(c['pbest']):.4g}"),
             f"robust rms = {1e3 * robust_sigma(r):.2f} mmag",
             f"fold P = {P:.6g} d,  binned p-p = {pp:.1f} mmag"]
        if an is not None:
            fs = "< 1e-16" if an[1] < 1e-16 else f"{an[1]:.2g}"
            L.append(f"FAP = {fs}  ({an[2].split(',')[0]}; white noise only!)")
        if self.rb_norm.value_selected.startswith("Bayes"):
            L.append(f"Bayes posterior σ_P = {self.bayes_local()['sP']:.3g} d (white, σ marginalised)")
        R = []
        if self.unc is not None:
            R.append(f"σ_P = {self.unc['sigma']:.3g} d  ({self.unc['method']}, n={self.unc['n']})")
        if self.red is not None:
            d = self.red
            R += [f"red-noise null: α = {d['alpha']:.3g} ({d['how']})",
                  f"  rms_red {1e3*d['rms_red']:.2f}, σ_white {1e3*d['sw']:.2f} mmag",
                  f"  99% max power {d['lev99']:.4f} (std)",
                  f"  p(max ≥ peak) = {d['pval']:.2g}  [{d['nsim']} sims]"]
        if self.boot is not None:
            R.append("σ_P comparison [d]:")
            R += [f"  {name:17s}{sg:9.4g}" for name, sg, n in self.boot]
        row = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), file=os.path.basename(self.name),
                   mjd_start=round(self.sel[0], 2), mjd_end=round(self.sel[1], 2), exclude=s["exclude"],
                   N=c["N"], nights=nights.size, pts_per_night=float(np.median(nights)), T_d=round(c["T"], 2),
                   detrend=s["detrend"], clip5sig=s["clip"], orbit_sub_P=s["orbit"] if s["orbit"] else np.nan,
                   K=c["K"], normalisation=self.rb_norm.value_selected, Pmin=c["pmin"], Pmax=c["pmax"],
                   P_best=Pb, f_best=c["fbest"], power_std=c["pbest"],
                   power_norm=np.nan if self.rb_norm.value_selected == "Bayes post" else self.nrm(c["pbest"]),
                   FAP=an[1] if an else np.nan, FAP_method=an[2] if an else "none",
                   sigma_P=self.unc["sigma"] if self.unc else np.nan,
                   sigma_P_method=self.unc["method"] if self.unc else "",
                   sigma_P_bayes=self.bayes_local()["sP"],
                   P2_over_T=Pb ** 2 / c["T"], rms_mmag=1e3 * robust_sigma(r), P_fold=P, profile_pp_mmag=pp,
                   red_alpha=self.red["alpha"] if self.red else np.nan,
                   red_maxpow99=self.red["lev99"] if self.red else np.nan,
                   red_pvalue=self.red["pval"] if self.red else np.nan)
        return L, R, row

    def redraw(self):
        for p in self.exc_patches:
            p.remove()
        self.exc_patches = [self.ax_lc.axvspan(a, b, color="0.5", alpha=0.35) for a, b in self.exclusions()]
        for a in (self.ax_dt, self.ax_ls, self.ax_fd, self.ax_tx):
            a.cla()
        self.ax_tx.axis("off")
        if self.cache is None:
            self.fig.canvas.draw_idle(); return
        self.draw_dt(self.ax_dt)
        self.draw_ls(self.ax_ls)
        self.draw_fold(self.ax_fd)
        L, R, _ = self.summary()
        self.ax_tx.text(0, 1.05, "\n".join(L), va="top", family="monospace", fontsize=8,
                        transform=self.ax_tx.transAxes)
        self.ax_tx.text(0.62, 1.05, "\n".join(R), va="top", family="monospace", fontsize=8,
                        transform=self.ax_tx.transAxes)
        if self.stale:
            self.status.set_text("settings changed: press [update]")
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- saving (edit 6)
    def stem(self):
        base = os.path.splitext(os.path.basename(self.name))[0]
        return os.path.join(self.outdir, f"{base}_{self.sel[0]:.0f}-{self.sel[1]:.0f}_K{self.cache['K']}")

    def save_panels(self):
        if self.cache is None:
            return
        os.makedirs(self.outdir, exist_ok=True)
        stem = self.stem(); out = []
        for key, fn, size in [("lightcurve", self.draw_lc, (10, 3.5)), ("detrended", self.draw_dt, (7, 4)),
                              ("periodogram", self.draw_ls, (7, 4.5)), ("folded", self.draw_fold, (7, 4))]:
            fig = Figure(figsize=size)
            ax = fig.add_subplot()
            fn(ax)
            fig.tight_layout()
            fig.savefig(f"{stem}_{key}.png", dpi=150)
            out.append(f"{stem}_{key}.png")
        self.fig.savefig(f"{stem}_full.png", dpi=120)
        L, R, _ = self.summary()
        with open(f"{stem}_summary.txt", "w") as fh:
            fh.write("\n".join(L + [""] + R) + "\n")
        print("[save panels] " + ", ".join(os.path.basename(p) for p in out) + " + _full.png, _summary.txt")
        self.set_status(f"saved panels: {stem}_*.png")

    def save_row(self):
        if self.cache is None:
            return
        os.makedirs(self.outdir, exist_ok=True)
        _, _, row = self.summary()
        csv_path = os.path.join(self.outdir, "explorer_results.csv")
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RESULT_COLS)
            if new:
                w.writeheader()
            w.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in row.items()})
        try:                                                # ECSV mirror (keeps dtypes for astropy users)
            Table.read(csv_path, format="ascii.csv").write(csv_path[:-4] + ".ecsv", format="ascii.ecsv",
                                                           overwrite=True)
        except Exception as err:                            # noqa: BLE001
            print("[save row] ECSV mirror not written:", err)
        print(f"[save row] appended to {csv_path}")
        self.set_status(f"row appended to {csv_path}")


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
    ex = Explorer(a.file, a.pmin, a.pmax, a.porb, workdir=a.dir, outdir=a.outdir, live=a.live, exclude=a.exclude, alias=a.alias)
    plt.show()


if __name__ == "__main__":
    main()
