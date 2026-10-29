#!/usr/bin/env python3
"""
Plot the conserved quantities and field energies of a NEOPIC run and its
reference (PIC, PIF or BH) from the npz files written by eval.py.

The files are written by ``save_diagnostics`` in eval.py, one for the NEOPIC
prediction and one for the reference:

    <evalDir>/diagnostics/pred_<testCase>_<ref>_N<nParticle>_run<runId>.npz
    <evalDir>/diagnostics/ref_<testCase>_<ref>_N<nParticle>_run<runId>.npz

Each file contains the time series (one value per time step)

    energy, Ek, Ep, momentum, Ex, Ey, Ez
    (cyclotron also: Lcan, Rrms, EerrPIC; Penning trap also: Eext)

where components that do not exist (Ey, Ez in lower dimensions, momentum for
the cyclotron case) are simply absent, and the metadata

    kind ('pred' or 'ref'), ref, nParticle, dim, testCase, dt, T, kc, alpha,
    runId, time_acc_ms (mean acceleration time per step in ms), and t_train
    (end of the training time horizon) if eval.py was given --tTrain

Usage
-----
    python plot_diagnostics.py --pred diagnostics/pred_weakLandau_pic_N200000_run1.npz \
                               --ref  diagnostics/ref_weakLandau_pic_N200000_run1.npz \
                               --out figures --ext pdf

Either --pred or --ref may be omitted. All plotting functions can also be
imported and called with dictionaries (this is what eval.py does).
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ---------------------------------------------------------------------------
# Style: same fonts and line widths as the error-scaling figures of the paper
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "figure.figsize": (7.5, 5.5),
}


def _styled(func):
    """Apply PLOT_STYLE locally, so other rcParams settings (e.g. those of
    pic_plotter.PICVisualizer) do not change these figures."""
    import functools

    @functools.wraps(func)
    def wrapper(*a, **k):
        with plt.rc_context(PLOT_STYLE):
            return func(*a, **k)
    return wrapper


# Colours identify the method, as in the performance figures of the paper:
# PIC blue, PIF orange, Barnes-Hut grey, NEOPIC red.
METHOD_COLORS = {"pic": "tab:blue", "pif": "tab:orange", "bh": "tab:gray",
                 "neopic": "tab:red"}
METHOD_NAMES = {"pic": "PIC", "pif": "PIF", "bh": "Barnes-Hut", "neopic": "NEOPIC"}
THEORY_STYLE = dict(color="k", linestyle="-.", linewidth=1.5)
# Relative changes that are exactly zero (always the first step, and steps where
# a quantity is bitwise identical to its initial value) cannot be shown on a log
# axis. They are drawn at this floor (double-precision machine epsilon), so the
# curves stay continuous; set to None to leave gaps instead.
ZERO_FLOOR = np.finfo(np.float64).eps
# End of the training time horizon: same style as the k = 0.5 line of the
# wave-number study and the training-horizon line of the mini-app study
TRAIN_LINE_STYLE = dict(color="0.5", linestyle=":", linewidth=1.5)

# Line styles identify the quantity within one figure
ENERGY_STYLES = {"energy": "-", "Ek": "--", "Ep": ":", "Eext": "-."}
ENERGY_LABELS = {"energy": "Total", "Ek": "Kinetic", "Ep": "Potential",
                 "Eext": "External potential"}
COMPONENT_STYLES = {"Ex": "-", "Ey": "--", "Ez": ":"}
COMPONENT_LABELS = {"Ex": r"$x$", "Ey": r"$y$", "Ez": r"$z$"}

TIME_LABEL = r"$t\,\omega_p$"

# Linear theory for the field energy W = int E^2 dx, W ~ exp(rate * t), with
# rate = 2 * gamma. The values are for k = 0.5 (bump-on-tail: k = 0.21); the
# theory line is skipped for other wave numbers. 'anchor' is the time at which
# the theory line is matched to the data.
THEORY = {
    "weakLandau": [dict(rate=-0.3066, anchor=2.5, label="Linear damping")],
    "strongLandau": [dict(rate=-0.562, anchor=2.5, label="Linear damping"),
                     dict(rate=0.168, anchor=20.592, label="Growth")],
    "tsi": [dict(rate=0.4952, anchor=8.0, label="Linear growth")],
    "bti": [dict(rate=0.356, anchor=8.0, label="Linear growth")],
}
# Wave number for which the rates above hold, per test case (default THEORY_K).
# The theory line is skipped for any other wave number.
THEORY_K = 0.5
THEORY_K_CASE = {"bti": 0.21}

# Default y-limits of the field-energy plots (from the original pic_plotter.py)
FIELD_YLIM = {
    (1, "strongLandau"): (1e-5, 10), (1, "tsi"): (1e-4, 1e2), (1, "bti"): (1e-4, 1e1),
    (2, "strongLandau"): (1e-3, None), (2, "weakLandau"): (1e-4, None),
    (2, "tsi"): (1e-4, 1e3), (2, "bti"): (None, 1e2),
    (3, "strongLandau"): (1e-3, 1e4), (3, "weakLandau"): (1e-3, 20),
    (3, "tsi"): (None, 1e4), (3, "bti"): (None, 1e4),
}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def load(fname):
    """Load an npz file written by eval.py into a plain dict."""
    if fname is None:
        return None
    with np.load(fname, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    for k, v in d.items():
        if v.ndim == 0:                       # metadata -> python scalars/strings
            d[k] = v.item()
    return d


def _series(d, key):
    """Time series `key` of record d, or None if absent or empty."""
    if d is None or key not in d:
        return None
    y = np.asarray(d[key], dtype=float)
    return y if y.size > 0 else None


def _time(d, n):
    return np.arange(n) * float(d.get("dt", 1.0))


def _meta(pred, ref, key, default=None):
    for d in (pred, ref):
        if d is not None and key in d:
            return d[key]
    return default


def _sources(pred, ref):
    """(record, colour, name) for each available record; reference first."""
    out = []
    if ref is not None:
        m = str(ref.get("ref", "pic")).lower()
        out.append((ref, METHOD_COLORS.get(m, "tab:blue"), METHOD_NAMES.get(m, m.upper())))
    if pred is not None:
        out.append((pred, METHOD_COLORS["neopic"], METHOD_NAMES["neopic"]))
    return out


def _legends(ax, sources, styles, style_labels, extra=None, loc_src="upper right",
             loc_sty="lower left"):
    """One legend for the methods (colours) and one for the quantities (line styles)."""
    h_src = [Line2D([], [], color=c, linestyle="-") for _, c, _ in sources]
    l_src = [n for _, _, n in sources]
    if extra:
        for h, l in extra:
            h_src.append(h)
            l_src.append(l)
    leg1 = ax.legend(h_src, l_src, loc=loc_src)
    if len(styles) > 1:
        h_sty = [Line2D([], [], color="0.3", linestyle=s) for s in styles.values()]
        ax.legend(h_sty, [style_labels[k] for k in styles], loc=loc_sty)
        ax.add_artist(leg1)


def _t_train(pred, ref, t_train=None):
    """Training horizon: explicit value, else the t_train metadata, else None."""
    if t_train is not None:
        return float(t_train)
    v = _meta(pred, ref, "t_train", None)
    return None if v is None else float(v)


def _train_line(axes, t_train):
    """Draw the end of the training horizon on every axis; return a legend entry."""
    if t_train is None:
        return []
    for ax in axes:
        h = ax.axvline(t_train, **TRAIN_LINE_STYLE)
    return [(h, "End of training")]


def _finish(fig, ax, fname, xlabel=TIME_LABEL, ylabel=None, tmax=None, rect=None):
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if tmax is not None:
        ax.set_xlim(0, tmax)
    ax.grid(True, which="both", alpha=0.3)
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    os.makedirs(os.path.dirname(os.path.abspath(fname)), exist_ok=True)
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"saved {fname}")
    return os.path.basename(fname)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _lw(name):
    """Reference drawn wider underneath, NEOPIC thinner on top, so both stay visible."""
    return (1.6, 3) if name == METHOD_NAMES["neopic"] else (2.6, 2)


@_styled
def plot_energy(pred, ref, fname, normalize=False, tmax=None, t_train=None):
    """
    Energy budget of the reference and NEOPIC. Top: total and kinetic energy
    (linear axis); bottom: potential (field) energy (log axis).
    """
    sources = _sources(pred, ref)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7.5, 7.5),
                                   gridspec_kw=dict(height_ratios=[1, 1.2]))
    used = {}
    for d, c, name in sources:
        e = _series(d, "energy")
        scale = e[0] if (normalize and e is not None) else 1.0
        lw, z = _lw(name)
        for q, ax in (("energy", ax1), ("Ek", ax1), ("Eext", ax1), ("Ep", ax2)):
            y = _series(d, q)
            if y is None:
                continue
            ls = ENERGY_STYLES[q] if q != "Ep" else "-"
            ax.plot(_time(d, len(y)), y / scale, color=c, linestyle=ls, linewidth=lw, zorder=z)
            used[q] = ls
    if not used:
        plt.close(fig)
        return None
    # log axis only when the potential energy varies over a decade or more (e.g. Landau
    # damping); a nearly constant one (cyclotron, Penning) is clearer on a linear axis
    ep = [y for d, _, _ in sources for y in [_series(d, "Ep")] if y is not None]
    ep = np.concatenate(ep) if ep else np.array([])
    ep = ep[np.isfinite(ep)]
    if ep.size and ep.min() > 0 and ep.max() / ep.min() >= 10:
        ax2.set_yscale("log")
    else:
        ax2.grid(True, which="both", alpha=0.3)
    styles = {q: ENERGY_STYLES[q] for q in ("energy", "Ek", "Eext") if q in used}
    extra = _train_line((ax1, ax2), _t_train(pred, ref, t_train))
    # one legend above the panels, so that no curve is covered:
    # methods (colours), end of training, quantities of the top panel (line styles)
    handles = [Line2D([], [], color=c, linestyle="-") for _, c, _ in sources]
    labels = [nm for _, _, nm in sources]
    for h, lab in extra:
        handles.append(h)
        labels.append(lab)
    for q, ls in styles.items():
        handles.append(Line2D([], [], color="0.3", linestyle=ls))
        labels.append(ENERGY_LABELS[q])
    ncol = 3 if len(handles) > 4 else len(handles)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    top = 0.90 if len(handles) > ncol else 0.94
    unit = r"$\,/\,\mathcal{E}(0)$" if normalize else ""
    ax1.set_ylabel("Energy" + unit)
    ax2.set_ylabel("Potential energy" + unit)
    ax1.grid(True, which="both", alpha=0.3)
    return _finish(fig, ax2, fname, tmax=tmax, rect=(0, 0, 1, top))


def _rel_change(y, absolute=False):
    d = np.abs(y - y[0])
    return d if absolute else d / np.abs(y[0])


@_styled
def plot_conservation(pred, ref, fname, momentum_absolute=False, tmax=None, t_train=None):
    """Relative change of total energy and of the momentum magnitude."""
    sources = _sources(pred, ref)
    styles = {"energy": "-", "momentum": "--"}
    if _meta(pred, ref, "testCase", "") == "penning":
        # the external electric and magnetic fields exert forces: momentum is not conserved
        styles = {"energy": "-"}
    if any(d is not None and "Lcan" in d for d in (pred, ref)):
        # cyclotron: no momentum series, but the canonical angular momentum is conserved
        styles = {"energy": "-", "Lcan": "--"}
    labels = {"energy": r"Energy $|\mathcal{E}-\mathcal{E}(0)|/|\mathcal{E}(0)|$",
              "Lcan": r"Angular momentum $|L-L(0)|/|L(0)|$",
              "momentum": (r"Momentum $|P-P(0)|$" if momentum_absolute
                           else r"Momentum $|P-P(0)|/|P(0)|$")}
    fig, ax = plt.subplots()
    used = {}
    for d, c, name in sources:
        for q, ls in styles.items():
            y = _series(d, q)
            if y is None:
                continue
            r = _rel_change(y, absolute=(q == "momentum" and momentum_absolute))
            if ZERO_FLOOR is None:
                r[r <= 0] = np.nan                # gaps where the change is exactly zero
            else:
                r = np.maximum(r, ZERO_FLOOR)     # exact zeros drawn at machine epsilon
            lw, z = _lw(name)
            ax.plot(_time(d, len(y)), r, color=c, linestyle=ls, linewidth=lw, zorder=z)
            used[q] = ls
    if not used:
        plt.close(fig)
        return None
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 100)                   # headroom for the legends
    extra = _train_line((ax,), _t_train(pred, ref, t_train))
    _legends(ax, sources, used, labels, extra=extra, loc_src="upper right",
             loc_sty="upper left")
    return _finish(fig, ax, fname, ylabel="Relative change", tmax=tmax)


def _theory_lines(test_case, t, anchor_data, kc):
    """Linear-theory curves matched to anchor_data at the anchor times."""
    lines = []
    if test_case not in THEORY:
        return lines
    k_th = THEORY_K_CASE.get(test_case, THEORY_K)
    if kc is not None and abs(float(kc) - k_th) > 1e-12:
        print(f"    theory rates for {test_case} are for k = {k_th}; skipped for k = {kc}")
        return lines
    for th in THEORY[test_case]:
        i = int(np.argmin(np.abs(t - th["anchor"])))
        if i >= len(anchor_data) or not np.isfinite(anchor_data[i]):
            continue
        y = anchor_data[i] * np.exp(th["rate"] * (t - t[i]))
        lines.append((y, rf"{th['label']} ($\propto e^{{{th['rate']:.3g}\,t}}$)"))
    return lines


@_styled
def plot_field_energy(pred, ref, fname, tmax=None, ylim="auto", theory=True, t_train=None):
    """
    Field energy per component, W_i = int E_i^2 dx, for Landau damping and
    instabilities, with the linear-theory rate where available.
    """
    sources = _sources(pred, ref)
    test_case = _meta(pred, ref, "testCase", "")
    dim = int(_meta(pred, ref, "dim", 1))
    kc = _meta(pred, ref, "kc", None)
    fig, ax = plt.subplots()
    used = {}
    for d, c, name in sources:
        lw, z = _lw(name)
        for q, ls in COMPONENT_STYLES.items():
            y = _series(d, q)
            if y is None:
                continue
            ax.plot(_time(d, len(y)), y, color=c, linestyle=ls, linewidth=lw, zorder=z)
            used[q] = ls
    if not used:
        plt.close(fig)
        return None

    extra = []
    if theory:
        # Match the theory to the NEOPIC x-component if present, as before;
        # for the instabilities use the highest available component.
        order = ("Ex",) if "Landau" in test_case else ("Ez", "Ey", "Ex")
        anchor = None
        for d in ([pred, ref] if "Landau" in test_case else [ref, pred]):
            for q in order:
                anchor = _series(d, q)
                if anchor is not None:
                    t_anchor = _time(d, len(anchor))
                    break
            if anchor is not None:
                break
        if anchor is not None:
            for k, (y, lab) in enumerate(_theory_lines(test_case, t_anchor, anchor, kc)):
                st = dict(THEORY_STYLE)
                if k == 1:
                    st["linestyle"] = (0, (5, 2, 1, 2, 1, 2))
                h, = ax.plot(t_anchor, y, **st)
                extra.append((h, lab))

    ax.set_yscale("log")
    if ylim == "auto":
        ylim = FIELD_YLIM.get((dim, test_case))
    if ylim is not None:
        ax.set_ylim(*ylim)
    extra += _train_line((ax,), _t_train(pred, ref, t_train))
    _legends(ax, sources, used, {k: COMPONENT_LABELS[k] for k in used}, extra=extra,
             loc_sty="lower left" if "Landau" in test_case else "lower right")
    ylabel = (r"Field energy $W_i=\int E_i^2\,d\mathbf{x}$" if len(used) > 1
              else r"Field energy $W_x=\int E_x^2\,d\mathbf{x}$")
    return _finish(fig, ax, fname, ylabel=ylabel, tmax=tmax)


@_styled
def plot_cyclotron(pred, ref, fname, tmax=None, t_train=None):
    """
    Cyclotron: rms radius of the cloud relative to t = 0 (top; it carries the
    magnetic part of the canonical angular momentum) and the relative error of
    each method's field against the free-space PIC field evaluated at that
    method's own positions (bottom, log axis).
    """
    sources = _sources(pred, ref)
    have_e = any(_series(d, "EerrPIC") is not None for d, _, _ in sources)
    have_r = any(_series(d, "Rrms") is not None for d, _, _ in sources)
    if not (have_e or have_r):
        return None
    n = int(have_e) + int(have_r)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(7.5, 3.6 * n + 0.6), squeeze=False)
    axes = list(axes[:, 0])
    k = 0
    if have_r:
        ax = axes[k]; k += 1
        for d, c, name in sources:
            y = _series(d, "Rrms")
            if y is None:
                continue
            lw, z = _lw(name)
            ax.plot(_time(d, len(y)), y / y[0], color=c, linewidth=lw, zorder=z)
        ax.set_ylabel(r"$r_{\mathrm{rms}}/r_{\mathrm{rms}}(0)$")
        ax.grid(True, which="both", alpha=0.3)
    if have_e:
        ax = axes[k]
        for d, c, name in sources:
            y = _series(d, "EerrPIC")
            if y is None:
                continue
            lw, z = _lw(name)
            ax.plot(_time(d, len(y)), y, color=c, linewidth=lw, zorder=z)
        ax.set_yscale("log")
        ax.set_ylabel(r"$\|E-E_{\mathrm{PIC}}\|/\|E_{\mathrm{PIC}}\|$")
    extra = _train_line(axes, _t_train(pred, ref, t_train))
    handles = [Line2D([], [], color=c) for _, c, _ in sources] + [h for h, _ in extra]
    labels = [nm for _, _, nm in sources] + [l for _, l in extra]
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    return _finish(fig, axes[-1], fname, tmax=tmax, rect=(0, 0, 1, 1 - 0.5 / (3.6 * n + 0.6)))


# ---------------------------------------------------------------------------
# Cyclotron snapshots (binned on a fixed grid by pic_plotter.cyclotron_snapshot)
# ---------------------------------------------------------------------------
# Fixed colour scales, the same for every method and step, so that snapshots can
# be compared directly. Density is n/N per unit area (integrates to 1).
SNAP_RHO_RANGE = (0.5, 200.0)
SNAP_EMAX = 20.0
SNAP_ERR_RANGE = (1e-3, 1.0)
SNAP_XLABEL, SNAP_YLABEL = r"$x$", r"$y$"


def _snap_time(rec):
    """Snapshot time as text, with as many decimals as the time step needs and no
    trailing zeros (0, 4, 8, 11.98 for dt = 0.01), so e.g. step 1198 is not
    rounded up to 12."""
    t, it = float(rec["t"]), int(rec["it"])
    dt = t / it if it > 0 else 0.0
    d = int(np.ceil(-np.log10(dt) - 1e-9)) if 0 < dt < 1 else 0
    return f"{t:.{max(d, 0)}f}".rstrip("0").rstrip(".") if d > 0 else f"{t:.0f}"


def _snap_name(rec):
    lab = str(rec.get("label", ""))
    return METHOD_NAMES.get(lab, lab.upper())


def _snap_maps(rec, rho_range=None, emax=None):
    """Arrays ([iy, ix] for imshow) and colour settings of one snapshot record."""
    ext = float(rec["extent"]); cnt = np.asarray(rec["counts"], dtype=float)
    nb = cnt.shape[0]; dA = (2 * ext / nb) ** 2
    empty = cnt == 0
    rho = np.where(empty, np.nan, cnt / (float(rec["nParticle"]) * dA))
    lo, hi = rho_range if rho_range is not None else SNAP_RHO_RANGE
    em = SNAP_EMAX if emax is None else float(emax)
    maps = {"rho": (rho, dict(cmap="inferno", norm=matplotlib.colors.LogNorm(lo, hi)),
                    r"density $n/N$"),
            "Ex": (np.where(empty, np.nan, rec["Ex"]), dict(cmap="coolwarm", vmin=-em, vmax=em),
                   r"$E_x$"),
            "Ey": (np.where(empty, np.nan, rec["Ey"]), dict(cmap="coolwarm", vmin=-em, vmax=em),
                   r"$E_y$")}
    if "dE" in rec:
        maps["dE"] = (np.where(empty, np.nan, np.maximum(rec["dE"], SNAP_ERR_RANGE[0])),
                      dict(cmap="viridis", norm=matplotlib.colors.LogNorm(*SNAP_ERR_RANGE)),
                      r"$|E-E_{\mathrm{PIC}}|/\max|E_{\mathrm{PIC}}|$")
    return maps, ext


def _nice_up(x):
    """Round up to 1, 2 or 5 times a power of ten."""
    if not np.isfinite(x) or x <= 0:
        return x
    e = 10.0 ** np.floor(np.log10(x))
    return next(m * e for m in (1, 2, 5, 10) if m * e >= x)


def _nice_down(x):
    if not np.isfinite(x) or x <= 0:
        return x
    e = 10.0 ** np.floor(np.log10(x))
    return next(m * e for m in (5, 2, 1) if m * e <= x)


def snapshot_check(rec, rho_range=None, emax=None, verbose=True):
    """
    Data range of one snapshot against the colour scales: occupied-bin density
    range, max |E_x|, |E_y|, max relative field error, and the fraction of the
    particles outside the snapshot box (these were not binned, so only a rerun
    with a larger --snapExtent can show them). Returns a dict; 'clip' lists what
    is clipped.
    """
    maps, _ = _snap_maps(rec, rho_range, emax)
    rho = maps["rho"][0]
    lo, hi = maps["rho"][1]["norm"].vmin, maps["rho"][1]["norm"].vmax
    em = maps["Ex"][1]["vmax"]
    info = dict(rho_min=float(np.nanmin(rho)), rho_max=float(np.nanmax(rho)),
                E_max=float(max(np.nanmax(np.abs(maps["Ex"][0])), np.nanmax(np.abs(maps["Ey"][0])))),
                outside=1.0 - float(np.sum(rec["counts"])) / float(rec["nParticle"]))
    clip = []
    if info["rho_max"] > hi:
        clip.append(f"density max {info['rho_max']:.3g} > {hi:g}")
    if info["rho_min"] < lo:
        clip.append(f"density min {info['rho_min']:.3g} < {lo:g}")
    if info["E_max"] > em:
        clip.append(f"|E| max {info['E_max']:.3g} > {em:g}")
    if "dE" in rec:
        info["dE_max"] = float(np.nanmax(rec["dE"]))
        if info["dE_max"] > SNAP_ERR_RANGE[1]:
            clip.append(f"field error max {info['dE_max']:.3g} > {SNAP_ERR_RANGE[1]:g}")
    info["clip"] = clip
    if verbose:
        tag = f"{rec.get('label', '')} it={int(rec['it'])}"
        note = ""
        if info["outside"] > 0:
            note = (f"  [{100 * info['outside']:.3g}% of the particles lie outside the box; "
                    f"only a rerun with a larger --snapExtent shows them]")
        print(f"snapshot {tag}: " + ("ok" if not clip else "CLIPPED: " + "; ".join(clip)) + note)
    return info


def _extend(norm_or_lims, data):
    """'neither'/'min'/'max'/'both' for the colour bar, from the data range."""
    lo, hi = norm_or_lims
    d = np.asarray(data, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return "neither"
    under, over = d.min() < lo, d.max() > hi
    return {(False, False): "neither", (True, False): "min",
            (False, True): "max", (True, True): "both"}[(under, over)]


def _lims(kw):
    n = kw.get("norm")
    return (n.vmin, n.vmax) if n is not None else (kw["vmin"], kw["vmax"])


def _snap_axis(ax, arr, kw, ext):
    cm = matplotlib.colormaps[kw.pop("cmap")].copy()
    cm.set_bad("white")
    return ax.imshow(np.asarray(arr).T, origin="lower", extent=[-ext, ext, -ext, ext],
                     cmap=cm, interpolation="nearest", aspect="equal", **kw)


@_styled
def plot_cyclotron_snapshot(rec, out_dir, tag="", ext="png", rho_range=None, emax=None, zoom=None):
    """Density, field (Ex, Ey) and, if present, field-error maps of one snapshot."""
    snapshot_check(rec, rho_range, emax)
    os.makedirs(out_dir, exist_ok=True)
    maps, L = _snap_maps(rec, rho_range, emax)
    lab, it = str(rec.get("label", "")), int(rec["it"])
    title = rf"{_snap_name(rec)}, $t\,\omega_p = {_snap_time(rec)}$"
    sfx = f"_{tag}" if tag else ""
    out = []
    groups = [("rho", ["rho"]), ("Efield", ["Ex", "Ey"])] + ([("Eerr", ["dE"])] if "dE" in maps else [])
    for gname, keys in groups:
        fig, axes = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 5.4), squeeze=False)
        for ax, key in zip(axes[0], keys):
            arr, kw, clab = maps[key]
            im = _snap_axis(ax, arr, dict(kw), L)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=clab,
                         extend=_extend(_lims(kw), arr))
            ax.set_xlabel(SNAP_XLABEL); ax.set_ylabel(SNAP_YLABEL)
            if zoom is not None:                  # show only [-zoom, zoom]^2 of the stored box
                ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
        fig.suptitle(title)
        fig.tight_layout()
        fname = os.path.join(out_dir, f"{gname}_{lab}_{it}{sfx}.{ext}")
        fig.savefig(fname, dpi=300); plt.close(fig)
        out.append(os.path.basename(fname))
    return out


@_styled
def plot_snapshot_grid(snap_dir, steps, quantity="rho", methods=("bh", "neopic"), run_id=1, zoom=None,
                       out=None, ext="png", rho_range=None, emax=None, t_train=None):
    """
    Paper figure from saved snapshots: one row per method, one column per step,
    one shared colour bar. quantity is rho, Ex, Ey or dE.
    """
    recs = {}
    for m in methods:
        for it in steps:
            f = os.path.join(snap_dir, f"snap_{m}_{it}_run{run_id}.npz")
            if os.path.exists(f):
                recs[m, it] = load(f)
    methods = [m for m in methods if any((m, it) in recs for it in steps)]
    if not methods:
        print(f"no snapshots found in {snap_dir}")
        return None
    fig, axes = plt.subplots(len(methods), len(steps), figsize=(3.3 * len(steps) + 1.2, 3.4 * len(methods)),
                             squeeze=False, sharex=True, sharey=True)
    im = None
    allvals = []
    for i, m in enumerate(methods):
        for j, it in enumerate(steps):
            ax = axes[i, j]
            rec = recs.get((m, it))
            if rec is None or (quantity == "dE" and "dE" not in rec):
                ax.set_axis_off()
                continue
            snapshot_check(rec, rho_range, emax)
            maps, L = _snap_maps(rec, rho_range, emax)
            arr, kw, clab = maps[quantity]
            im = _snap_axis(ax, arr, dict(kw), L)
            if zoom is not None:
                ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
            allvals.append(np.ravel(arr)); lims = _lims(kw)
            t = float(rec["t"])
            if i == 0:
                mark = "  (extrap.)" if (t_train is not None and t > t_train) else ""
                ax.set_title(rf"$t\,\omega_p = {_snap_time(rec)}${mark}", fontsize=13)
            if j == 0:
                ax.set_ylabel(f"{_snap_name(rec)}\n{SNAP_YLABEL}")
            if i == len(methods) - 1:
                ax.set_xlabel(SNAP_XLABEL)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    if im is not None:
        cax = fig.add_axes([0.915, 0.12, 0.015, 0.76])
        fig.colorbar(im, cax=cax, label=clab, extend=_extend(lims, np.concatenate(allvals)))
    out = out or os.path.join(snap_dir, f"grid_{quantity}_run{run_id}.{ext}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"saved {out}")
    return os.path.basename(out)


def spiral_m2(rec, rmax=None, dr=0.005):
    """
    m = 2 (two-arm) azimuthal Fourier analysis of a snapshot density, about the
    density centroid. For each radius bin: amplitude |A_2|/A_0 and arm angle
    phi(r) = -arg(A_2)/2 (defined mod pi, unwrapped in r), with
    A_m(r) = sum_bins n e^{-i m theta}. Returns (r, A_0, amplitude, phi).
    """
    c = np.asarray(rec["counts"], dtype=float); ext = float(rec["extent"]); nb = c.shape[0]
    xc = (np.arange(nb) + 0.5) / nb * 2 * ext - ext
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    w = c / c.sum(); X = X - (w * X).sum(); Y = Y - (w * Y).sum()
    R = np.hypot(X, Y); T = np.arctan2(Y, X)
    rmax = ext if rmax is None else rmax
    edges = np.arange(0.0, rmax + dr, dr); rc = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(R.ravel(), edges) - 1; ok = (idx >= 0) & (idx < len(rc))
    cw = c.ravel()[ok]; ii = idx[ok]; tt = T.ravel()[ok]
    A0 = np.bincount(ii, cw, len(rc))
    A2 = np.bincount(ii, cw * np.cos(2 * tt), len(rc)) - 1j * np.bincount(ii, cw * np.sin(2 * tt), len(rc))
    amp = np.abs(A2) / np.maximum(A0, 1.0)
    phi = np.unwrap(-np.angle(A2)) / 2
    return rc, A0, amp, phi


def spiral_winding(rec, r_in=0.09, r_out=0.29):
    """Winding of the two-arm spiral between r_in and r_out in turns,
    [phi(r_out) - phi(r_in)] / 2 pi, and phi(r) on that range."""
    rc, _, _, phi = spiral_m2(rec, rmax=r_out + 0.01)
    sel = (rc >= r_in) & (rc <= r_out)
    p = np.unwrap(2 * phi[sel]) / 2
    return (p[-1] - p[0]) / (2 * np.pi), rc[sel], p


@_styled
def plot_winding(snap_dir, methods=("bh", "neopic"), run_id=1, r_in=0.09, r_out=0.29,
                 out=None, ext="png", t_train=None, profile_steps=None):
    """
    Spiral winding from the saved snapshots. Top: winding number (turns of the
    arms between r_in and r_out) against time for each method. Bottom: arm angle
    phi(r) - phi(r_in) at the given steps (default: the last one), the shape of the arms.
    Also prints the winding table and writes it to winding_run<id>.csv.
    """
    import glob, re
    data = {}
    for m in methods:
        for f in glob.glob(os.path.join(snap_dir, f"snap_{m}_*_run{run_id}.npz")):
            it = int(re.search(rf"snap_{m}_(\d+)_run", os.path.basename(f)).group(1))
            data.setdefault(m, {})[it] = load(f)
    methods = [m for m in methods if m in data]
    if not methods:
        print(f"no snapshots in {snap_dir}")
        return None
    steps = sorted(set.intersection(*[set(data[m]) for m in methods]))
    rows = []
    wind = {m: [] for m in methods}
    times = []
    for it in steps:
        t = float(data[methods[0]][it]["t"]); times.append(t)
        for m in methods:
            wind[m].append(spiral_winding(data[m][it], r_in, r_out)[0])
        rows.append([it, t] + [wind[m][-1] for m in methods])
    hdr = "step,t," + ",".join(f"winding_{m}" for m in methods)
    csv = os.path.join(os.path.dirname(out) if out else snap_dir, f"winding_run{run_id}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(csv)), exist_ok=True)
    np.savetxt(csv, np.array(rows), delimiter=",", header=hdr, comments="", fmt="%.6g")
    print(hdr); [print(",".join(f"{v:.4g}" for v in r)) for r in rows]

    profile_steps = profile_steps or [steps[-1]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8.2))
    for m in methods:
        c = METHOD_COLORS.get(m, "k"); name = METHOD_NAMES.get(m, m)
        lw, z = _lw(name)
        ax1.plot(times, np.abs(wind[m]), "o-", color=c, linewidth=lw, zorder=z, markersize=4)
        for k, it in enumerate(profile_steps):
            if it not in data[m]:
                continue
            _, r, p = spiral_winding(data[m][it], r_in, r_out)
            ax2.plot(r, np.degrees(np.abs(p - p[0])), color=c, linewidth=lw, zorder=z,
                     linestyle=["-", "--", ":", "-."][k % 4])
    extra = _train_line((ax1,), _t_train(None, None, t_train))
    ax1.set_xlabel(TIME_LABEL)
    ax1.set_ylabel(rf"winding, $r\in[{r_in:g},{r_out:g}]$ [turns]")
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$|\varphi(r)-\varphi(r_{\mathrm{in}})|$ [deg]")
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
    handles = [Line2D([], [], color=METHOD_COLORS.get(m, "k")) for m in methods] + [h for h, _ in extra]
    labels = [METHOD_NAMES.get(m, m) for m in methods] + [l for _, l in extra]
    if len(profile_steps) > 1:
        for k, it in enumerate(profile_steps):
            handles.append(Line2D([], [], color="0.3", linestyle=["-", "--", ":", "-."][k % 4]))
            labels.append(rf"$t\,\omega_p={_snap_time(data[methods[0]][it])}$")
    else:
        ax2.set_title(rf"arm shape at $t\,\omega_p={_snap_time(data[methods[0]][profile_steps[0]])}$",
                      fontsize=13)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 4), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out or os.path.join(snap_dir, f"winding_run{run_id}.{ext}")
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"saved {out}")
    return os.path.basename(out)


def check_snapshots(snap_dir, rho_range=None, emax=None, run_id=None):
    """
    Check every snap_*.npz in snap_dir against the colour scales and print the
    ranges that would cover all of them (to pass as --snap-rho-range/--snap-emax,
    or as --snapRhoRange/--snapEmax in eval.py). Nothing is plotted.
    """
    import glob
    pat = f"snap_*_run{run_id}.npz" if run_id is not None else "snap_*.npz"
    files = sorted(glob.glob(os.path.join(snap_dir, pat)))
    if not files:
        print(f"no snapshots in {snap_dir}")
        return None
    infos = [snapshot_check(load(f), rho_range, emax) for f in files]
    rmin = min(i["rho_min"] for i in infos); rmax = max(i["rho_max"] for i in infos)
    emx = max(i["E_max"] for i in infos)
    n_clip = sum(bool(i["clip"]) for i in infos)
    out = max(i["outside"] for i in infos)
    print(f"{n_clip}/{len(infos)} snapshots clipped. Data: density {rmin:.3g}..{rmax:.3g}, "
          f"max |E| {emx:.3g}, max outside the box {100 * out:.3g}%")
    print(f"suggested: --snap-rho-range {_nice_down(rmin):g} {_nice_up(rmax):g} "
          f"--snap-emax {_nice_up(emx):g}")
    return dict(rho=(rmin, rmax), emax=emx, outside=out, n_clipped=n_clip)


def plot_all(pred, ref, out_dir, tag="", ext="png", **kw):
    """
    Make every figure that applies to the test case. Returns a dict of file
    names (keys: energy, conservation, field) for the evaluation summary.
    File names follow the original pic_plotter.py names.
    """
    test_case = _meta(pred, ref, "testCase", "")
    sfx = f"_{tag}" if tag else ""
    out = {"energy": plot_energy(pred, ref, os.path.join(out_dir, f"Energy{sfx}.{ext}"),
                                 normalize=kw.get("normalize", False), tmax=kw.get("tmax"),
                                 t_train=kw.get("t_train"))}
    out["conservation"] = None
    if test_case != "penning" or "Eext" in (pred or ref or {}):
        out["conservation"] = plot_conservation(
            pred, ref, os.path.join(out_dir, f"conservation_errors{sfx}.{ext}"),
            momentum_absolute=kw.get("momentum_absolute", False), tmax=kw.get("tmax"),
            t_train=kw.get("t_train"))
    out["field"] = None
    if test_case in ("weakLandau", "strongLandau"):
        out["field"] = plot_field_energy(
            pred, ref, os.path.join(out_dir, f"landau_decay_rateRef{sfx}.{ext}"),
            tmax=kw.get("tmax"), ylim=kw.get("ylim", "auto"), t_train=kw.get("t_train"))
    elif test_case in ("tsi", "bti"):
        out["field"] = plot_field_energy(
            pred, ref, os.path.join(out_dir, f"growth_rate{sfx}.{ext}"),
            tmax=kw.get("tmax"), ylim=kw.get("ylim", "auto"), t_train=kw.get("t_train"))
    elif test_case == "cyclotron":
        out["field"] = plot_cyclotron(
            pred, ref, os.path.join(out_dir, f"cyclotron_radius_field_error{sfx}.{ext}"),
            tmax=kw.get("tmax"), t_train=kw.get("t_train"))
    return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=None, help="npz file of the NEOPIC prediction")
    ap.add_argument("--ref", default=None, help="npz file of the reference")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--tag", default=None,
                    help="suffix of the figure names (default: from the metadata)")
    ap.add_argument("--ext", default="png", help="figure format, e.g. png or pdf")
    ap.add_argument("--tmax", type=float, default=None, help="upper time limit")
    ap.add_argument("--ymin", type=float, default=None, help="lower y-limit of field plots")
    ap.add_argument("--ymax", type=float, default=None, help="upper y-limit of field plots")
    ap.add_argument("--t-train", type=float, default=None,
                    help="end of the training horizon in t omega_p, drawn as a dotted line "
                         "(default: the t_train metadata of the files, if any)")
    ap.add_argument("--normalize", action="store_true",
                    help="divide the energies by the initial total energy")
    ap.add_argument("--momentum-absolute", action="store_true",
                    help="plot |P-P(0)| instead of |P-P(0)|/|P(0)|")
    ap.add_argument("--snap-dir", default=None,
                    help="cyclotron: directory with snap_<method>_<it>_run<id>.npz files; makes "
                         "comparison grids (rows: methods, columns: --snap-steps)")
    ap.add_argument("--snap-steps", type=int, nargs="+", default=[0, 400, 800, 1199])
    ap.add_argument("--snap-methods", nargs="+", default=["bh", "neopic"])
    ap.add_argument("--snap-quantities", nargs="+", default=["rho", "Ex", "dE"])
    ap.add_argument("--snap-run", type=int, default=1)
    ap.add_argument("--snap-rho-range", type=float, nargs=2, default=None)
    ap.add_argument("--snap-emax", type=float, default=None)
    ap.add_argument("--snap-winding", action="store_true",
                    help="cyclotron: winding of the two-arm spiral from the snapshots (figure + csv)")
    ap.add_argument("--wind-r", type=float, nargs=2, default=[0.09, 0.29],
                    help="radial range of the winding measurement (outside the core)")
    ap.add_argument("--snap-zoom", type=float, default=None,
                    help="show only [-zoom, zoom]^2 of the stored snapshot box (no rerun needed)")
    ap.add_argument("--snap-check", action="store_true",
                    help="only check the snapshots in --snap-dir against the colour scales "
                         "and print ranges that cover all of them")
    ap.add_argument("--snap-single", action="store_true",
                    help="also redraw the per-step snapshot figures (rho_, Efield_, Eerr_)")
    args = ap.parse_args()
    if args.snap_dir is not None and args.snap_winding:
        plot_winding(args.snap_dir, methods=args.snap_methods, run_id=args.snap_run,
                     r_in=args.wind_r[0], r_out=args.wind_r[1], ext=args.ext, t_train=args.t_train,
                     out=os.path.join(args.out, f"winding_run{args.snap_run}.{args.ext}"))
        return
    if args.snap_dir is not None and args.snap_check:
        check_snapshots(args.snap_dir, args.snap_rho_range, args.snap_emax, run_id=args.snap_run)
        return
    if args.snap_dir is not None and args.snap_single:
        import glob
        for f in sorted(glob.glob(os.path.join(args.snap_dir, f"snap_*_run{args.snap_run}.npz"))):
            plot_cyclotron_snapshot(load(f), args.out, tag=f"run{args.snap_run}", ext=args.ext,
                                    rho_range=args.snap_rho_range, emax=args.snap_emax,
                                    zoom=args.snap_zoom)
    if args.snap_dir is not None:
        for q in args.snap_quantities:
            plot_snapshot_grid(args.snap_dir, args.snap_steps, quantity=q, methods=args.snap_methods,
                               run_id=args.snap_run, ext=args.ext, rho_range=args.snap_rho_range,
                               emax=args.snap_emax, t_train=args.t_train, zoom=args.snap_zoom,
                               out=os.path.join(args.out, f"grid_{q}_run{args.snap_run}.{args.ext}"))
        if args.pred is None and args.ref is None:
            return
    if args.pred is None and args.ref is None:
        ap.error("give --pred and/or --ref")

    pred, ref = load(args.pred), load(args.ref)
    if args.tag is None:
        d = ref if ref is not None else pred
        args.tag = f"{d.get('testCase', '')}_{d.get('ref', '')}_N{d.get('nParticle', '')}"
    ylim = "auto"
    if args.ymin is not None or args.ymax is not None:
        ylim = (args.ymin, args.ymax)
    plot_all(pred, ref, args.out, tag=args.tag, ext=args.ext, tmax=args.tmax, ylim=ylim,
             normalize=args.normalize, momentum_absolute=args.momentum_absolute,
             t_train=args.t_train)


if __name__ == "__main__":
    main()
