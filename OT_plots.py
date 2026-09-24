"""ot_fit — standalone library for OT displacement-profile fitting & diagnostics.

Self-contained: model (double_ag), flexible per-parameter bounds, goodness-of-fit
in physical kpc and as %R200c, and CSV-driven plots that work on any results table
with the canonical columns. No dependency on Func_for_fit / profiles HDF5 layout
except the optional HDF5-overlay plot, which is opt-in.

Canonical parameter order:
    C_pos, mu_pos, wL_pos, wR_pos, S0_pos, C_neg, mu_neg, w_neg
Canonical CSV columns (produced by fit_dataframe):
    snap, sim, lo, hi, logM, a, z, x10, n_halos, n_pts,
    R200c_pkpc, rms_kpc, rms_pct_r200, chi2_dof, fit_ok, <param>, <param>_err ...
"""
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

PNAMES = ["C_pos", "mu_pos", "wL_pos", "wR_pos", "S0_pos", "C_neg", "mu_neg", "w_neg"]

# ----------------------------------------------------------------- model --------
def r200c_pkpc(dmo_r200_med_cMpc, a):
    """Median DMO r200c (comoving Mpc) -> physical kpc. r=1000*cMpc*a."""
    return 1000.0 * dmo_r200_med_cMpc * a

def asym_gauss(lnr, C, mu, wL, wR, S0):
    """Asymmetric Gaussian in ln r. Width interpolates geometrically wL<->wR via
    tanh((lnr-mu)/|S0|): left tail ~wL, right tail ~wR, S0 sets transition sharpness."""
    t = np.tanh((lnr - mu) / abs(S0))
    w = abs(wL) ** ((1 - t) / 2) * abs(wR) ** ((1 + t) / 2)
    return C * np.exp(-0.5 * ((lnr - mu) / w) ** 2)

def sym_gauss(lnr, C, mu, w):
    return C * np.exp(-0.5 * ((lnr - mu) / abs(w)) ** 2)

def double_ag(r, C_pos, mu_pos, wL_pos, wR_pos, S0_pos, C_neg, mu_neg, w_neg):
    """Positive asymmetric peak minus negative symmetric dip, in ln r."""
    lnr = np.log(r)
    return (asym_gauss(lnr, C_pos, mu_pos, wL_pos, wR_pos, S0_pos)
            - sym_gauss(lnr, abs(C_neg), mu_neg, w_neg))

def load(csv):
    """Read a results CSV (str path) or pass a DataFrame through unchanged."""
    return csv if isinstance(csv, pd.DataFrame) else pd.read_csv(csv) 

# ----------------------------------------------------------------- bounds -------
# default (lo, hi) per parameter; mu_pos used as ceiling for mu_neg unless overridden
DEFAULT_BOUNDS = {
    "C_pos":  (0.0, 20.0), "mu_pos": (-5.0, 5.0),
    "wL_pos": (0.01, 3.0), "wR_pos": (0.01, 3.0), "S0_pos": (0.01, 5.0),
    "C_neg":  (0.0, 20.0), "mu_neg": (-5.0, None),  # hi=None -> tie to mu_pos init
    "w_neg":  (0.01, 3.0),
}

def _build_bounds(bounds, mp0):
    """Merge user bounds over defaults -> (lo[], hi[]). hi=None on mu_neg ties to mp0."""
    b = {**DEFAULT_BOUNDS, **(bounds or {})}
    lo, hi = [], []
    for p in PNAMES:
        l, h = b[p]
        if h is None: h = mp0 if p == "mu_neg" else np.inf
        lo.append(l); hi.append(h)
    return np.array(lo, float), np.array(hi, float)


# ----------------------------------------------------------------- fitting ------
def fit_one(rc, ot, dmo_r200_med, a, sigma_kpc=10.0, bounds=None,
            p0=None, maxfev=20000, meta=None):
    """Fit one OT map. rc, ot in R200c units (y=ot-rc fitted). dmo_r200_med in cMpc.
    bounds: dict {param:(lo,hi)} overriding DEFAULT_BOUNDS; mu_neg hi=None ties to mu_pos init.
    Returns flat dict (one CSV row): params, *_err, gof, fit_ok, + any meta keys."""
    rc = np.asarray(rc, float); y = np.asarray(ot, float) - rc
    R = r200c_pkpc(dmo_r200_med, a)
    ip, ig = int(np.argmax(y)), int(np.argmin(y))
    mp0, mn0 = np.log(rc[ip]), np.log(rc[ig])
    if mn0 >= mp0: mn0 = mp0 - 0.5
    if p0 is None:
        p0 = [y.max(), mp0, 0.5, 0.2, 0.3, abs(y.min()), mn0, 0.3]
    lo, hi = _build_bounds(bounds, mp0)
    p0 = np.clip(p0, lo, hi)                     # keep init inside bounds
    row = dict(meta or {}); row["R200c_pkpc"] = R; row["n_pts"] = len(rc)
    try:
        sg = np.full_like(rc, sigma_kpc / R)
        popt, pcov = curve_fit(double_ag, rc, y, p0=p0, bounds=(lo, hi),
                               sigma=sg, absolute_sigma=True, maxfev=maxfev)
        perr = np.sqrt(np.diag(pcov))
        row.update(gof(rc, y, popt, R, sigma_kpc)); row["fit_ok"] = True
    except (RuntimeError, ValueError):
        popt = perr = [np.nan] * len(PNAMES)
        row.update(rms_kpc=np.nan, rms_pct_r200=np.nan, chi2_dof=np.nan); row["fit_ok"] = False
    for nm, v, e in zip(PNAMES, popt, perr): row[nm] = v; row[f"{nm}_err"] = e
    return row

def fit_dataframe(targets, sigma_kpc=10.0, bounds=None, out_csv=None,
                  key_cols=("snap", "sim", "lo", "hi"), flush_every=50, verbose=True):
    """Fit a list of target dicts -> DataFrame. Resume-safe if out_csv exists.
    Each target needs: rc, ot, dmo_r200_med, a, + meta (snap,sim,lo,hi,logM,z,x10,n_halos...).
    Meta keys present on the target (minus rc/ot/dmo_r200_med arrays) are copied to the row."""
    import os
    arr = {"rc", "ot", "avg_dmo", "avg_hyd", "edges", "dmo_r200_med"}
    done, rows = set(), []
    if out_csv and os.path.exists(out_csv):
        prev = pd.read_csv(out_csv); rows = prev.to_dict("records")
        done = set(map(tuple, prev[list(key_cols)].itertuples(index=False, name=None)))
        if verbose: print(f"resume: {len(done)} done, {len(targets)-len(done)} to go")
    todo = [t for t in targets if tuple(t.get(k) for k in key_cols) not in done]
    for i, t in enumerate(todo, 1):
        meta = {k: v for k, v in t.items() if k not in arr}
        meta.setdefault("logM", 0.5 * (t["lo"] + t["hi"]) if {"lo", "hi"} <= t.keys() else np.nan)
        meta.setdefault("z", 1 / t["a"] - 1)
        r = fit_one(t["rc"], t["ot"], t["dmo_r200_med"], t["a"],
                    sigma_kpc=sigma_kpc, bounds=bounds, meta=meta)
        rows.append(r)
        if verbose:
            print(f"[{i}/{len(todo)}] {r.get('snap')}/{r.get('sim')} "
                  f"chi2/dof={r['chi2_dof']:.2f} RMS={r['rms_kpc']:.1f}kpc"
                  f"{'' if r['fit_ok'] else ' FAIL'}")
        if out_csv and (i % flush_every == 0 or i == len(todo)):
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    if out_csv: df.to_csv(out_csv, index=False)
    if verbose and len(df):
        print(f"-> total={len(df)} ok={int(df.fit_ok.sum())} fail={int((~df.fit_ok).sum())}")
    return df

# ----------------------------------------------------------------- plot utils ---
def _filter(df, x10_max=None, chi2_max=None, sims=None, ok_only=True):
    if ok_only and "fit_ok" in df: df = df[df.fit_ok]
    if x10_max is not None and "x10" in df: df = df[df.x10 < x10_max]
    if chi2_max is not None and "chi2_dof" in df: df = df[df.chi2_dof < chi2_max]
    if sims is not None: df = df[df.sim.isin(sims)]
    return df

def _sim_cmap(sims):
    import matplotlib.cm as cm
    c = cm.get_cmap("tab10", max(len(sims), 1))
    return {s: c(i) for i, s in enumerate(sims)}

def load(csv):
    """Read a results CSV (str path) or pass through a DataFrame."""
    return csv if isinstance(csv, pd.DataFrame) else pd.read_csv(csv)

# ----------------------------------------------------------------- plots --------
def plot_params_vs_mass(csv, params=None, sims=None, x10_max=None, chi2_max=None):
    """Grid: rows=params, cols=sims. x=logM, color=z."""
    import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.colors as mcolors
    df = _filter(load(csv), x10_max, chi2_max, sims)
    if not len(df): print("nothing passes filters"); return
    P = params or [p for p in PNAMES if p in df]; sims = sims or sorted(df.sim.unique())
    norm = mcolors.Normalize(df.z.min(), df.z.max())
    fig, ax = plt.subplots(len(P), len(sims), figsize=(2.2*len(sims), 1.6*len(P)),
                           sharex=True, sharey="row", squeeze=False)
    for j, s in enumerate(sims):
        d = df[df.sim == s]
        for i, p in enumerate(P):
            a = ax[i, j]
            a.scatter(d.logM, d[p], c=d.z, cmap="viridis", norm=norm, s=8, alpha=.7, ec="none")
            if j == 0: a.set_ylabel(p, fontsize=8)
            if i == 0: a.set_title(s, fontsize=8)
            if i == len(P)-1: a.set_xlabel(r"$\log M$")
            a.tick_params(labelsize=7)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="z", pad=.01)
    fig.suptitle(f"OT params (n={len(df)})"); plt.show()

def plot_fit_quality(csv, sims=None, chi2_vmax=10, rms_vmax=50):
    """Two rows (chi2/dof, rms_kpc) vs (logM,z), one col per sim."""
    import matplotlib.pyplot as plt
    df = _filter(load(csv), sims=sims); sims = sims or sorted(df.sim.unique())
    fig, ax = plt.subplots(2, len(sims), figsize=(2.4*len(sims), 5),
                           sharex=True, sharey="row", squeeze=False)
    for j, s in enumerate(sims):
        d = df[df.sim == s]
        for i, (col, vmax) in enumerate([("chi2_dof", chi2_vmax), ("rms_kpc", rms_vmax)]):
            sc = ax[i, j].scatter(d.logM, d.z, c=d[col], s=10, cmap="magma_r",
                                  vmin=0, vmax=vmax, ec="none")
            if j == 0: ax[i, j].set_ylabel(f"z\n[{col}]")
            if i == 0: ax[i, j].set_title(s, fontsize=8)
            if i == 1: ax[i, j].set_xlabel(r"$\log M$")
            plt.colorbar(sc, ax=ax[i, j], fraction=.05)
    plt.tight_layout(); plt.show()

def plot_param_triangle(csv, color_by="logM", sims=None, x10_max=0.3, chi2_max=5,
                        params=None, bins=30):
    """Corner plot of params; lower=scatter w/ Pearson r, diag=hist."""
    import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.colors as mcolors
    df = _filter(load(csv), x10_max, chi2_max, sims)
    if not len(df): print("nothing passes filters"); return
    P = params or [p for p in PNAMES if p in df]; n = len(P)
    c = df[color_by] if color_by else None
    norm = mcolors.Normalize(c.min(), c.max()) if color_by else None
    fig, ax = plt.subplots(n, n, figsize=(1.9*n, 1.9*n), squeeze=False)
    for i in range(n):
        for j in range(n):
            a = ax[i, j]
            if j > i: a.axis("off"); continue
            if i == j:
                a.hist(df[P[i]], bins=bins, color="0.4", ec="none"); a.set_yticks([])
            else:
                if color_by:
                    a.scatter(df[P[j]], df[P[i]], c=c, cmap="viridis", norm=norm, s=5, alpha=.5, ec="none")
                else:
                    a.scatter(df[P[j]], df[P[i]], s=5, alpha=.3, color="k", ec="none")
                r = np.corrcoef(df[P[j]], df[P[i]])[0, 1]
                a.text(.05, .92, f"{r:+.2f}", transform=a.transAxes, fontsize=7,
                       color="firebrick" if abs(r) > .5 else "0.3",
                       fontweight="bold" if abs(r) > .5 else "normal")
            if i == n-1: a.set_xlabel(P[j], fontsize=8)
            else: a.set_xticklabels([])
            if j == 0: a.set_ylabel(P[i], fontsize=8)
            else: a.set_yticklabels([])
            a.tick_params(labelsize=6)
    if color_by:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label=color_by, pad=.01, fraction=.03)
    fig.suptitle(f"param corner (n={len(df)})", y=.92); plt.show()

def plot_fits_overlay(csv, h5, sims=None, x10_max=None, chi2_max=None, every=1,
                      logM=None, z=None, mtol=0.15, ztol=0.05):
    """OT data (dots) + fitted double_ag (line) per sim, colored by logM.
    Requires h5 with groups '{snap}/{sim}/{lo:.3f}_{hi:.3f}' containing rc, ot.
    logM, z: scalar or list of target values to select (nearest within mtol/ztol).
             None = no constraint on that axis (original behaviour)."""
    import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.colors as mcolors, h5py
    df = _filter(load(csv), x10_max, chi2_max, sims)
    sel = lambda col, vals, tol: np.any([np.abs(df[col]-v) <= tol for v in np.atleast_1d(vals)], axis=0)
    if logM is not None: df = df[sel("logM", logM, mtol)]
    if z is not None:    df = df[sel("z", z, ztol)]
    if every > 1: df = df.iloc[::every]
    sims = sims or sorted(df.sim.unique())
    if not len(df): print("no rows match selection"); return
    norm = mcolors.Normalize(df.logM.min(), df.logM.max())
    with h5py.File(h5, "r") as f:
        for s in sims:
            d = df[df.sim == s]
            if not len(d): continue
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.axhline(0, color="k", ls="--", lw=1); ax.axvline(1, color="gray", ls=":", lw=1)
            for _, r in d.iterrows():
                g = f[f"{int(r.snap)}/{r.sim}/{r.lo:.3f}_{r.hi:.3f}"]
                rc = g["rc"][:]; y = g["ot"][:] - rc; col = cm.plasma(norm(r.logM))
                ax.plot(rc, y, "o", ms=2, alpha=.4, mec="none", color=col)
                rs = np.geomspace(rc.min(), rc.max(), 200)
                ax.plot(rs, double_ag(rs, *[r[p] for p in PNAMES]), "-", lw=.6, alpha=.5, color=col)
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap="plasma"), ax=ax, label=r"$\log M$")
            ax.set(xscale="log", xlabel=r"$r/R_{200c}$", ylabel=r"$T(r)-r$ [$R_{200c}$]",
                   title=f"{s} ({len(d)} maps)")
            plt.tight_layout(); plt.show() 

def fit_targets(out_h5="profiles_v3.hdf5"):
    """Maps passing the fit gate: status=='ok' AND no empty bins anywhere on either side.
    Returns full grid + provenance for fitting."""
    out=[]
    import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.colors as mcolors, h5py
    with h5py.File(out_h5,"r") as f:
        for snap in sorted((k for k in f.keys() if k!="meta"),key=int):
            for sim in f[snap]:
                for k in f[f"{snap}/{sim}"]:
                    g=f[f"{snap}/{sim}/{k}"]; a_=g.attrs
                    if "ot" not in g: continue
                    if a_["status"]!="ok": continue
                    if a_["n_empty_dmo"]!=0 or a_["n_empty_hyd"]!=0: continue
                    out.append(dict(
                        snap=int(snap),sim=sim,
                        lo=float(a_["lo"]),hi=float(a_["hi"]),
                        a=float(a_["a"]),x10=float(a_["x10"]),
                        dmo_r200_med=float(a_["dmo_r200_med"]),       # cMpc
                        n_halos=int(a_["n_halos"]),
                        rc=g["rc"][:],ot=g["ot"][:],
                        avg_dmo=g["avg_dmo"][:],avg_hyd=g["avg_hyd"][:],
                        edges=g["edges"][:]))
    return out

targets=fit_targets()
print(f"{len(targets)} maps pass") 


def gof(rc, y, popt, R200c, sigma_kpc):
    """Goodness of fit. rc in R200c units, y=T(r)-r (R200c units), R200c in pkpc.
    Returns dict: rms_kpc, rms_pct_r200, chi2_dof, n_pts."""
    res = y - double_ag(rc, *popt)
    rms_kpc = float(np.sqrt(np.mean((res * R200c) ** 2)))
    sg = sigma_kpc / R200c                      # noise floor in R200c units
    dof = max(len(rc) - len(popt), 1)
    chi2 = float(np.sum((res / sg) ** 2) / dof)
    return dict(rms_kpc=rms_kpc, rms_pct_r200=100.0 * rms_kpc / R200c,
                chi2_dof=chi2, n_pts=len(rc)) 


def load(csv):
    """Read a results CSV (str path) or pass through a DataFrame."""
    return csv if isinstance(csv, pd.DataFrame) else pd.read_csv(csv)
 
def _key(row):
    """HDF5 group key for a results row: '{snap}/{sim}/{lo:.3f}_{hi:.3f}'."""
    return f"{int(row['snap'])}/{row['sim']}/{row['lo']:.3f}_{row['hi']:.3f}"
 
def load_map(h5, row):
    """Read the real (rc, ot) grid for one results row from profiles HDF5.
    h5: path or open h5py.File/Group. row: a DataFrame row (Series) or dict
    with snap, sim, lo, hi. Returns (rc, ot) arrays in r/R200c units."""
    import h5py
    f = h5py.File(h5, "r") if isinstance(h5, str) else h5
    try:
        g = f[_key(row)]
        return g["rc"][:], g["ot"][:]
    finally:
        if isinstance(h5, str): f.close()
 
def eval_fit(row, rc):
    """double_ag evaluated on grid rc using the fitted params in row (Series/dict)."""
    return double_ag(rc, *[row[p] for p in PNAMES])
 
def gof_row(h5, row, sigma_kpc=10.0):
    """Recompute GOF for one results row against its real map. Returns gof dict.
    Uses R200c_pkpc straight from the row (already stored in v4 CSV)."""
    rc, ot = load_map(h5, row)
    return gof(rc, ot - rc, [row[p] for p in PNAMES],
               R200c=row["R200c_pkpc"], sigma_kpc=sigma_kpc)
  

def plot_param_triangle(csv="ot_fit_v3.csv", color_by="logM", sims=None,
                        x10_max=0.3, chi2_max=5, params=None, bins=30):
    import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.colors as mcolors, h5py 
    """Corner plot of the 8 fit params vs each other. Lower triangle = scatter colored
    by color_by ('logM','z', or None for density); diagonal = 1D histograms.
    Filters mirror plot_params_vs_mass: x10_max, chi2_max, fit_ok."""
    df=pd.read_csv(csv)
    if x10_max is not None: df=df[df.x10<x10_max]
    if chi2_max is not None: df=df[df.chi2_dof<chi2_max]
    df=df[df.fit_ok]
    if sims is not None: df=df[df.sim.isin(sims)]
    P=params or PNAMES; n=len(P)
    if len(df)==0: print("nothing passes filters"); return

    c=df[color_by] if color_by else None
    norm=mcolors.Normalize(c.min(),c.max()) if color_by else None
    cmap=cm.viridis
    fig,axes=plt.subplots(n,n,figsize=(1.9*n,1.9*n))
    for i in range(n):
        for j in range(n):
            ax=axes[i,j]
            if j>i: ax.axis("off"); continue                       # upper triangle blank
            if i==j:                                                # diagonal: 1D hist
                ax.hist(df[P[i]],bins=bins,color="0.4",ec="none")
                ax.set_yticks([])
            else:                                                   # lower: scatter
                if color_by:
                    ax.scatter(df[P[j]],df[P[i]],c=c,cmap=cmap,norm=norm,
                               s=5,alpha=0.5,ec="none")
                else:
                    ax.scatter(df[P[j]],df[P[i]],s=5,alpha=0.3,color="k",ec="none")
                r=np.corrcoef(df[P[j]],df[P[i]])[0,1]               # Pearson r
                ax.text(0.05,0.92,f"{r:+.2f}",transform=ax.transAxes,
                        fontsize=7,color="firebrick" if abs(r)>0.5 else "0.3",
                        fontweight="bold" if abs(r)>0.5 else "normal")
            if i==n-1: ax.set_xlabel(P[j],fontsize=8)
            else: ax.set_xticklabels([])
            if j==0 and i>0: ax.set_ylabel(P[i],fontsize=8)
            elif j==0: ax.set_ylabel(P[i],fontsize=8)              # diagonal top-left
            else: ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
    if color_by:
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,
                     label=color_by,pad=0.01,fraction=0.03)
    fig.suptitle(f"OT fit parameter corner  (n={len(df)}"
                 +(f", x10<{x10_max}" if x10_max else "")
                 +(f", χ²/dof<{chi2_max}" if chi2_max else "")+")",y=0.92)
    plt.show() 


def plot_params_vs_mass(csv="ot_fit_v3.csv", sims=None, x10_max=None, chi2_max=None):
    import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.colors as mcolors, h5py  
    """8 params × 1 row each, columns = sims. logM on x, color = z. Filters: x10_max, chi2_max."""
    df=pd.read_csv(csv)
    if x10_max is not None: df=df[df.x10<x10_max]
    if chi2_max is not None: df=df[df.chi2_dof<chi2_max]
    df=df[df.fit_ok]
    sims=sims or sorted(df.sim.unique())
    norm=mcolors.Normalize(df.z.min(),df.z.max()); cmap=cm.viridis
    fig,axes=plt.subplots(len(PNAMES),len(sims),figsize=(2.2*len(sims),1.6*len(PNAMES)),
                          sharex=True,sharey='row')
    for j,sim in enumerate(sims):
        d=df[df.sim==sim]
        for i,p in enumerate(PNAMES):
            ax=axes[i,j]
            ax.scatter(d.logM,d[p],c=d.z,cmap=cmap,norm=norm,s=8,alpha=0.7,ec='none')
            if j==0: ax.set_ylabel(p,fontsize=8)
            if i==0: ax.set_title(sim,fontsize=8)
            if i==len(PNAMES)-1: ax.set_xlabel(r'$\log M$')
            ax.tick_params(labelsize=7)
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,label='z',pad=0.01)
    fig.suptitle(f"OT fit parameters  (n={len(df)})"
                 +(f"  x10<{x10_max}" if x10_max else "")
                 +(f"  χ²/dof<{chi2_max}" if chi2_max else ""))
    plt.show()