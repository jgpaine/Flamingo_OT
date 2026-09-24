import numpy as np
#import illustris_python as il
#from illustris_python import snapshot
import matplotlib.pyplot as plt
#import h5py, os, glob, tarfile, time, requests
from pathlib import Path
from math import sqrt, pi
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.special import erf, erfcx, iv
import scipy.special as sc
from tqdm import tqdm

def T_ot_eval(x): return np.interp(x,be,T_ot_e)

# fit map callable (same as before, but as function on edges)
def T_fit_eval(x):
    x=np.asarray(x,float)
    y=np.interp(x,r_fit,T_fit)
    s0=T_fit[0]/r_fit[0]
    s1=(T_fit[-1]-T_fit[-2])/(r_fit[-1]-r_fit[-2]) if len(r_fit)>1 else 1.0
    y=np.where(x<r_fit[0], s0*x, y)
    y=np.where(x>r_fit[-1], T_fit[-1]+s1*(x-r_fit[-1]), y)
    return y 

def find_fof_halos(base, snap, Mmin=1e12, Mmax=1e13, n=100): 
    groups_dir = Path(base) / "output" / f"groups_{snap:03d}"
    results = []
    offset = 0
 
    for fname in sorted(os.listdir(groups_dir)):
        if not fname.endswith(".hdf5"):
            continue
 
        with h5py.File(groups_dir / fname, "r") as h5:
            # Always read h from header — never hardcode it
            h = h5["Header"].attrs["HubbleParam"]
 
            if "Group" not in h5 or "GroupMass" not in h5["Group"]:
                offset += 0  # no groups in this chunk — offset unchanged
                continue
 
            grp = h5["Group"]
            n_this_file = grp["GroupMass"].shape[0]
            M = grp["GroupMass"][:] * 1e10 / h  # Msun
 
            sel = np.where((M > Mmin) & (M < Mmax))[0]
            for i in sel:
                fof_id = offset + int(i)
                results.append({
                    "FoF_ID":           fof_id,
                    "M_FoF":            float(M[i]),
                    "R_crit200":        float(grp["Group_R_Crit200"][i] / h),  # ckpc
                    "CentralSubhaloID": int(grp["GroupFirstSub"][i]),
                    "FoF_Position":     (grp["GroupPos"][i] / h).astype(float),
                })
                if len(results) >= n:
                    return results
 
            offset += n_this_file  # advance by actual group count in this file
 
    return results


def find_fof_halos_old(path, snap, h=0.6774, Mmin=1e11, Mmax=1e12, n=10):
    import illustris_python as il
    import numpy as np
    from pathlib import Path

    # ensure path is a string
    path = str(path)

    # check if 'groups_###' exists; if not, try adding 'output/'
    import os
    grp_dir = os.path.join(path, f"groups_{snap:03d}")
    if not os.path.exists(grp_dir):
        path = os.path.join(path, "output")
        grp_dir = os.path.join(path, f"groups_{snap:03d}")
        if not os.path.exists(grp_dir):
            raise FileNotFoundError(f"Cannot find group catalog for snapshot {snap} in {path}")

    # fields to load
    f = ["GroupMass","Group_R_Crit200","GroupFirstSub","GroupPos"]
    g = il.groupcat.loadHalos(path, snap, fields=f)

    M = g["GroupMass"]*1e10/h
    ids = np.where((M>Mmin) & (M<Mmax))[0][:n]

    return [{"R_crit200": float(g["Group_R_Crit200"][i]/h),
             "FoF_ID": int(i),
             "M_FoF": float(M[i]),
             "CentralSubhaloID": int(g["GroupFirstSub"][i]),
             "FoF_Position": (g["GroupPos"][i]/h).astype(float)}
            for i in ids]

def load_match_context(baseHydro, baseDMO, snap):
    bh, bd = str(baseHydro/"output"), str(baseDMO/"output")
    mf = str(baseHydro/"postprocessing"/"released"/"subhalo_matching_to_dark.hdf5")

    Hh = il.groupcat.loadHalos(bh, snap, fields=["GroupFirstSub","GroupPos","GroupMass","Group_R_Crit200"])
    Hd = il.groupcat.loadHalos(bd, snap, fields=["GroupPos","GroupMass","Group_R_Crit200"])
    
    Sh = il.groupcat.loadSubhalos(bh, snap, fields=["SubhaloPos"])
    Sd = il.groupcat.loadSubhalos(bd, snap, fields=["SubhaloPos","SubhaloGrNr"])

    f = h5py.File(mf, "r")
    link = f[f"Snapshot_{snap}"]["SubhaloIndexDark_SubLink"]  

    return dict(Hh=Hh, Hd=Hd, Sh=Sh, Sd=Sd, link=link, h5=f) 

def psi_safe(r, alpha, a, b, c, sigma):
    """
    Numerically stable version of Matt's closed-form potential.
    Eliminates exp(+x) * exp(-x) overflow using erfcx.
    """
    r = np.asarray(r, dtype=float)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return np.full_like(r, np.nan)
    z = r / (np.sqrt(2.0) * sigma)
    x = z**2                      # = r^2 / (2 sigma^2)
    gaussian = np.exp(-x)
    poly_part = ( 6.0 * sigma * (b * r + 2.0 * c * sigma)  + 2.0 * a * (r**2 + 2.0 * sigma**2))
    erf_safe = 1.0 - gaussian * erfcx(z)
    erf_part = ( 3.0 * b * np.sqrt(2.0 * np.pi) * sigma**2  * erf_safe)
    psi = alpha * gaussian * (poly_part - erf_part) / (12.0 * sigma)
    return -psi

def make_hist_cdf_icdf(bin_edges, hist):
    # from discrete histogram get the CDF and inverse CDF 
    hist = np.asarray(hist, float)
    hist /= hist.sum()
    cdf_vals = np.concatenate([[0.0], np.cumsum(hist)])
    r_vals = np.asarray(bin_edges, float) # 
    # interpolate the so we can use funcations at any value of r 
    return (
        lambda r: np.interp(r, r_vals, cdf_vals, left=0.0, right=1.0), #CDF 
        lambda u: np.interp(u, cdf_vals, r_vals),   )

def ot_map_from_hists(bin_edges, hist_A, hist_B, r):
    # unique 1-D  optimal transport map
    cdf_A, _ = make_hist_cdf_icdf(bin_edges, hist_A)
    _, icdf_B = make_hist_cdf_icdf(bin_edges, hist_B)
    return icdf_B(cdf_A(r))
        
def fit_params_to_ot(bin_edges, hist_A, hist_B, r, psi_func=psi_safe):
    T_ot = ot_map_from_hists(bin_edges, hist_A, hist_B, r)
    res = lambda p: map_T(
        r, p[0], p[1], p[2], p[3], np.exp(p[4]),
        psi_func=psi_func
    ) - T_ot
    p = least_squares(res, [1, 0, 0, 0, 0]).x
    return p[0], p[1], p[2], p[3], np.exp(p[4])
        
def ot_map_from_hists(bin_edges, hist_A, hist_B, r):
    # unique 1-D  optimal transport map
    cdf_A, _ = make_hist_cdf_icdf(bin_edges, hist_A)
    _, icdf_B = make_hist_cdf_icdf(bin_edges, hist_B)
    return icdf_B(cdf_A(r)) 
        
def map_T(r, alpha, a, b, c, sigma, psi_func=psi_safe):
    psi = psi_func(r, alpha, a, b, c, sigma)
    psi -= psi[0]

    dT = np.exp(psi)
    T = np.zeros_like(r)
    T[1:] = np.cumsum(0.5 * (dT[1:] + dT[:-1]) * np.diff(r))

    return T * (r[-1] / T[-1]) 
def load_match_context(baseHydro, baseDMO, snap):
    bh, bd = str(baseHydro/"output"), str(baseDMO/"output")
    mf = str(baseHydro/"postprocessing"/"released"/"subhalo_matching_to_dark.hdf5")

    Hh = il.groupcat.loadHalos(bh, snap, fields=["GroupFirstSub","GroupPos","GroupMass","Group_R_Crit200"])
    Hd = il.groupcat.loadHalos(bd, snap, fields=["GroupPos","GroupMass","Group_R_Crit200"])
    
    Sh = il.groupcat.loadSubhalos(bh, snap, fields=["SubhaloPos"])
    Sd = il.groupcat.loadSubhalos(bd, snap, fields=["SubhaloPos","SubhaloGrNr"])

    f = h5py.File(mf, "r")
    link = f[f"Snapshot_{snap}"]["SubhaloIndexDark_SubLink"]  

    return dict(Hh=Hh, Hd=Hd, Sh=Sh, Sd=Sd, link=link, h5=f)

def fof_to_matched_dmo_info_fast(ctx, snap, fof_h):
    Hh, Hd, Sh, Sd, link = ctx["Hh"], ctx["Hd"], ctx["Sh"], ctx["Sd"], ctx["link"]

    sub_h = int(Hh["GroupFirstSub"][fof_h])
    cen_h = Sh["SubhaloPos"][sub_h] if isinstance(Sh, dict) else Sh[sub_h]  # safe if Sh is array-like

    sub_d = int(link[sub_h])
    fof_d = int(Sd["SubhaloGrNr"][sub_d])
    cen_d = Sd["SubhaloPos"][sub_d]

    return {
        "Hydro": dict(FoF=int(fof_h), CentralSubhalo=sub_h, Center=cen_h,
                     FoF_Position=Hh["GroupPos"][fof_h], M_FoF=Hh["GroupMass"][fof_h],
                     R_crit200=Hh["Group_R_Crit200"][fof_h]),
        "DMO":   dict(FoF=fof_d, Subhalo=sub_d, Center=cen_d,
                     FoF_Position=Hd["GroupPos"][fof_d], M_FoF=Hd["GroupMass"][fof_d],
                     R_crit200=Hd["Group_R_Crit200"][fof_d]),
    }

def load_fof_particles(basePath, snap, fof, types=("gas","dm","stars","bh")):
    basePath=str(basePath)
    snap=int(np.asarray(snap).reshape(-1)[0]); fof=int(np.asarray(fof).reshape(-1)[0])
    if not os.path.isdir(f"{basePath}/snapdir_{snap:03d}") and os.path.isdir(f"{basePath}/output/snapdir_{snap:03d}"):
        basePath=f"{basePath}/output"
    pt={"gas":"gas","dm":"dm","stars":"star","bh":"bh"}
    mdm=None
    if "dm" in types:
        g=sorted(glob.glob(f"{basePath}/snapdir_{snap:03d}/snap_{snap:03d}.*.hdf5"))
        if not g: raise FileNotFoundError(f"No snapshot chunks at {basePath}/snapdir_{snap:03d}/snap_{snap:03d}.*.hdf5")
        with h5py.File(g[0],"r") as H: mdm=float(H["Header"].attrs["MassTable"][1])
    out={}
    for k in types:
        try:
            fields=["Coordinates","ParticleIDs"] + ([] if k=="dm" else ["Masses"])
            D=il.snapshot.loadHalo(basePath, snap, fof, pt[k], fields=fields)
            if not D or "ParticleIDs" not in D or "Coordinates" not in D:
                out[k]=None; continue
            pid=np.asarray(D["ParticleIDs"])
            pos=np.asarray(D["Coordinates"],float)
            m=np.full(len(pid), mdm, float) if k=="dm" else np.asarray(D["Masses"],float)
            out[k]={"id":pid,"m":m,"pos":pos}
        except Exception:
            out[k]=None; continue
    return out 

''' 
def load_fof_particles(basePath, snap, fof, types=("gas","dm","stars","bh")):
    basePath=str(basePath)
    snap=int(np.asarray(snap).reshape(-1)[0]); fof=int(np.asarray(fof).reshape(-1)[0])
    if not os.path.isdir(f"{basePath}/snapdir_{snap:03d}") and os.path.isdir(f"     {basePath}/output/snapdir_{snap:03d}"):
        basePath=f"{basePath}/output"
    pt={"gas":"gas","dm":"dm","stars":"star","bh":"bh"}
    mdm=None
    if "dm" in types:
        g=sorted(glob.glob(f"{basePath}/snapdir_{snap:03d}/snap_{snap:03d}.*.hdf5"))
        if not g: raise FileNotFoundError(f"No snapshot chunks at {basePath}/snapdir_{snap:03d}/snap_{snap:03d}.*.hdf5")
        with h5py.File(g[0],"r") as H: mdm=float(H["Header"].attrs["MassTable"][1])
    out={}
    for k in types:
        fields=["Coordinates","ParticleIDs"] + ([] if k=="dm" else ["Masses"])
        D=il.snapshot.loadHalo(basePath, snap, fof, pt[k], fields=fields)
        if not D: out[k]=None; continue
        pid=np.asarray(D["ParticleIDs"])
        pos=np.asarray(D["Coordinates"],float)
        m=np.full(len(pid), mdm, float) if k=="dm" else np.asarray(D["Masses"],float)
        out[k]={"id":pid,"m":m,"pos":pos}
    return out 
''' 
def halo_vectors(ctx, baseHydro, baseDMO, snap, fof_h, types=("gas","dm","stars","bh")):
    m=fof_to_matched_dmo_info_fast(ctx,snap,fof_h); H,D=m["Hydro"],m["DMO"]
    bh,bd=baseHydro/"output",baseDMO/"output"
    return {"fof_h":H["FoF"],"cen_h":np.asarray(H["Center"],float),"M200c_h":float(H["M_FoF"]),"R200c_h":float(H["R_crit200"]),
            "fof_d":D["FoF"],"cen_d":np.asarray(D["Center"],float),"M200c_d":float(D["M_FoF"]),"R200c_d":float(D["R_crit200"]),
            "parts_h":load_fof_particles(bh,snap,H["FoF"],types=types),
            "parts_d":load_fof_particles(bd,snap,D["FoF"],types=("dm",))} 


def make_hist(ctx,baseHydro,baseDMO,snap,fof_h,r0,n,rmax=1e5):
    out=halo_vectors(ctx,baseHydro,baseDMO,snap,fof_h)
    edges=np.r_[0.,r0,np.logspace(np.log10(r0),np.log10(rmax),n+1)[1:]]
    def r(pos,cen): return np.linalg.norm(pos-np.asarray(cen)[None,:],axis=1)

    def sum_types(parts,cen,keys=("gas","dm","stars","bh")):
        M=np.zeros(len(edges)-1,float)
        for k in keys:
            v=parts.get(k)
            if v is None: continue
            M+=np.histogram(r(v["pos"],cen),bins=edges,weights=v["m"])[0]
            #M+=np.histogram(r(v["pos"],cen),bins=edges,weights=None)[0]
        return M
    Mh=sum_types(out["parts_h"],out["cen_h"])
    dmD=out["parts_d"]["dm"]
    Md=np.histogram(r(dmD["pos"],out["cen_d"]),bins=edges,weights=dmD["m"])[0]
    ''' 
    title=f"fof_h={out['fof_h']}  fof_d={out['fof_d']}"
    plt.figure()
    plt.step(edges[:-1],Mh,where="post",label="Hydro all (gas+dm+stars+bh)")
    plt.step(edges[:-1],Md,where="post",label="DMO dm")
    plt.xscale("log"); 
    #plt.yscale("log")
    plt.yscale("log"); 
    plt.xlabel("r (from each run center)"); plt.ylabel("mass per bin")
    plt.title(title); plt.legend(); plt.show()
    ''' 
    out["hist"]={"edges":edges,"Mh":Mh,"Md":Md,"r0":r0,"rmax":rmax,"n":n}
    return out 

# fit map callable (same as before, but as function on edges)
def T_fit_eval(x):
    x=np.asarray(x,float)
    y=np.interp(x,r_fit,T_fit)
    s0=T_fit[0]/r_fit[0]
    s1=(T_fit[-1]-T_fit[-2])/(r_fit[-1]-r_fit[-2]) if len(r_fit)>1 else 1.0
    y=np.where(x<r_fit[0], s0*x, y)
    y=np.where(x>r_fit[-1], T_fit[-1]+s1*(x-r_fit[-1]), y)
    return y 

def push_hist_monotone(be,m,Te):
    e=np.asarray(be,float); m=np.asarray(m,float)
    t=Te(e); t=np.maximum.accumulate(t)              # enforce monotone on edges
    out=np.zeros_like(m)
    for i in range(len(m)):
        if m[i]==0: continue
        a,b=t[i],t[i+1]
        if b<a: a,b=b,a
        if b==a:
            j=np.searchsorted(e,a,side="right")-1
            if 0<=j<len(out): out[j]+=m[i]
            continue
        j0=max(np.searchsorted(e,a,side="right")-1,0)
        j1=min(np.searchsorted(e,b,side="left"),len(out))
        for j in range(j0,j1):
            L=max(a,e[j]); R=min(b,e[j+1])
            if R>L: out[j]+=m[i]*(R-L)/(b-a)
    return out/out.sum() 


def shell_mass(profile_fn, bin_edges, n_points=300):
        masses = np.zeros(len(bin_edges) - 1)
        for i, (r1, r2) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            r_lo  = max(r1, 1e-7)   # avoid r = 0
            r_int = np.geomspace(r_lo, r2, n_points)
            rho   = profile_fn(r_int)
            masses[i] = np.trapz(4 * np.pi * r_int**2 * rho, r_int)
            return masses
        
def get_profiles(bp, cosmo, M_c, M200=None, a_sf=1.0, n_r200=5, plot=False):
    import pyccl as ccl
    import BaryonForge as bfg
    h = cosmo.cosmo.params.h
    R200    = ccl.halos.massdef.MassDef200c.get_radius(cosmo, M200, a_sf)
    r       = np.geomspace(0.005 * R200, n_r200 * R200, 500)
    rho_dmo = bfg.DarkMatterOnly(**bp).real(cosmo, r, M200, a_sf)
    rho_dmb = bfg.DarkMatterBaryon(M_c=M_c, **bp).real(cosmo, r, M200, a_sf)
    if plot:
        fig, ax = plt.subplots()
        ax.loglog(r*h, r**2 * rho_dmo, 'k:', lw=3, label='DMO')
        ax.loglog(r*h, r**2 * rho_dmb, lw=3,       label='DMB')
        ax.set(xlabel=r'$r$ [Mpc/h]', ylabel=r'$r^2\rho$')
        ax.legend(frameon=False)
        plt.show()


    return r, rho_dmo, rho_dmb

def get_histograms(rho_dmo, rho_dmb, r, cosmo, M200, a_sf=1.0, n_bins=150, plot=False):
    import pyccl as ccl
    R200      = ccl.halos.massdef.MassDef200c.get_radius(cosmo, M200, a_sf)
    bin_edges = np.geomspace(0.01 * R200, 5 * R200, n_bins + 1)

    def shell_mass(rho):
        M = np.zeros(len(bin_edges) - 1)
        for i, (r1, r2) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            mask = (r >= r1) & (r <= r2)
            if mask.sum() < 2: continue
            M[i] = np.trapz(4*np.pi * r[mask]**2 * rho[mask], r[mask])
        return M

    H_dmo, H_dmb = shell_mass(rho_dmo), shell_mass(rho_dmb)
    n_dmo, n_dmb = H_dmo / H_dmo.sum(), H_dmb / H_dmb.sum()

    if plot:
        be_plot = np.geomspace(0.01 * R200, 5 * R200, 31)
        rc_fine = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        rc_plot = 0.5 * (be_plot[:-1]   + be_plot[1:])
        H_dmo_p = np.interp(rc_plot, rc_fine, H_dmo)
        H_dmb_p = np.interp(rc_plot, rc_fine, H_dmb)
        fig, ax = plt.subplots()
        ax.step(be_plot[1:]*h, H_dmo_p, where='pre', color='steelblue', lw=2, label='DMO')
        ax.step(be_plot[1:]*h, H_dmb_p, where='pre', color='orange',    lw=2, label='DMB')
        ax.set(xscale='log', yscale='log', xlabel=r'$r$ [Mpc/h]', ylabel=r'Shell Mass [$M_\odot$]')
        ax.legend(frameon=False); plt.show()

    return n_dmo, n_dmb, bin_edges


def get_ot_map(n_dmo, n_dmb, bin_edges, plot=False):
    """Compute OT displacement map Δr vs r."""
    be = bin_edges
    rc = 0.5 * (be[:-1] + be[1:])
    T  = ot_map_from_hists(be, n_dmo, n_dmb, rc)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(rc, T - rc, 'o', ms=4, label=r'OT $\Delta r$')
        ax.axhline(0, color='k', ls='--', lw=1)
        ax.set(xscale='log', xlabel=r'$r$ [Mpc]', ylabel=r'$\Delta r$ [Mpc]')
        ax.legend(); plt.show()
    return rc, T


def fit_ot(rc, T, n_dmo, n_dmb, bin_edges, fit_fn=None, plot=False):
    """Fit OT map, return params dict and optionally plot."""
    be   = bin_edges
    mask = (rc >= be[1]) & np.isfinite(T) & (T > 0)
    r_fit = rc[mask]
    alpha, a, b, c, sigma = fit_params_to_ot(be, n_dmo, n_dmb, r_fit, psi_func=F.psi_safe)
    T_fit = np.maximum.accumulate(np.clip(
        map_T(r_fit, alpha, a, b, c, sigma, psi_func=F.psi_safe), 0., be[-1]))
    if plot:
        fig, ax = plt.subplots()
        ax.plot(rc,    T     - rc,    'o', ms=4, label='OT')
        ax.plot(r_fit, T_fit - r_fit, '-', lw=2, label='Fit')
        ax.axhline(0, color='k', ls='--', lw=1)
        ax.set(xscale='log', xlabel=r'$r$ [Mpc]', ylabel=r'$\Delta r$ [Mpc]')
        ax.legend(); plt.show()
    return dict(alpha=alpha, a=a, b=b, c=c, sigma=sigma, r_fit=r_fit, T_fit=T_fit)


def safe_fit(fn, r, d, p0, bounds):
    from scipy.optimize import curve_fit
    try:
        popt, _ = curve_fit(fn, r, d, p0=p0, bounds=bounds, maxfev=20000)
        return popt, np.sqrt(np.mean((fn(r, *popt) - d)**2)), True
    except Exception as e:
        print(f"  fit failed: {e}")
        return p0, np.nan, False