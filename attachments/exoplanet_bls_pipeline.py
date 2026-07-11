"""
SCO EXOPLANET HUNTER - Stage 1 Pipeline
Author: Aisha Shermatova
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk
import os

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "axes.grid": False,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#222222",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DATA_COLOR = "#3B5A82"
MODEL_COLOR = "#D9622B"
BIN_COLOR = "#111111"


def _extract_arrays(lc_flat):
    time = np.asarray(lc_flat.time.value, dtype=float)
    flux = np.asarray(lc_flat.flux.value, dtype=float)
    if getattr(lc_flat, "flux_err", None) is not None:
        flux_err = np.asarray(lc_flat.flux_err.value, dtype=float)
    else:
        flux_err = None
    good = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        good &= np.isfinite(flux_err)
        flux_err = flux_err[good]
    else:
        mad = np.nanmedian(np.abs(flux - np.nanmedian(flux)))
        flux_err = np.full(good.sum(), 1.4826 * mad)
    time = time[good]
    flux = flux[good]
    order = np.argsort(time)
    return time[order], flux[order], flux_err[order]


def run_bls(time, flux, flux_err, min_period=0.5, max_period=15.0,
            n_periods=30000, min_duration_hr=0.5, max_duration_hr=6.0,
            n_durations=15):
    bls = BoxLeastSquares(time, flux, dy=flux_err)
    periods = np.linspace(min_period, max_period, n_periods)
    durations = np.linspace(min_duration_hr / 24.0, max_duration_hr / 24.0, n_durations)
    result = bls.power(periods, durations, objective="snr")
    best_idx = int(np.argmax(result.power))
    best = {
        "period": float(result.period[best_idx]),
        "duration": float(result.duration[best_idx]),
        "t0": float(result.transit_time[best_idx]),
        "power": float(result.power[best_idx]),
        "depth": float(result.depth[best_idx]),
        "depth_err": float(result.depth_err[best_idx]),
        "depth_snr": float(result.depth_snr[best_idx]),
    }
    return bls, result, best


def period_uncertainty_from_peak(result, best_idx=None):
    periods = result.period
    power = result.power
    if best_idx is None:
        best_idx = int(np.argmax(power))
    peak_power = power[best_idx]
    half_max = 0.5 * (peak_power - np.median(power)) + np.median(power)
    left = best_idx
    while left > 0 and power[left] > half_max:
        left -= 1
    right = best_idx
    while right < len(power) - 1 and power[right] > half_max:
        right += 1
    return max((periods[right] - periods[left]) / 2.0, periods[1] - periods[0])


def bootstrap_errors(time, flux, flux_err, best, n_boot=200,
                     period_window_frac=0.05, block_size=50, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(time)
    n_blocks = int(np.ceil(n / block_size))
    lo = best["period"] * (1 - period_window_frac)
    hi = best["period"] * (1 + period_window_frac)
    boot_periods_grid = np.linspace(lo, hi, 400)
    durations = np.array([best["duration"]])
    boot_periods, boot_depths = [], []
    for _ in range(n_boot):
        block_starts = rng.integers(0, n_blocks, size=n_blocks) * block_size
        idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in block_starts])
        idx = idx[idx < n]
        if len(idx) < 10:
            continue
        t_b, f_b, e_b = time[idx], flux[idx], flux_err[idx]
        order = np.argsort(t_b)
        t_b, f_b, e_b = t_b[order], f_b[order], e_b[order]
        try:
            bls_b = BoxLeastSquares(t_b, f_b, dy=e_b)
            res_b = bls_b.power(boot_periods_grid, durations, objective="snr")
            j = int(np.argmax(res_b.power))
            boot_periods.append(res_b.period[j])
            boot_depths.append(res_b.depth[j])
        except Exception:
            continue
    boot_periods = np.array(boot_periods)
    boot_depths = np.array(boot_depths)
    return {
        "period_err_bootstrap": float(np.std(boot_periods)) if len(boot_periods) > 5 else np.nan,
        "depth_err_bootstrap": float(np.std(boot_depths)) if len(boot_depths) > 5 else np.nan,
        "n_successful_draws": len(boot_periods),
    }


def vet_candidate(bls, best, stats, snr_threshold=7.0,
                  depth_ppm_eb_threshold=30000.0,
                  duration_period_ratio_threshold=0.15):
    flags = []
    depth_odd, depth_odd_err = stats["depth_odd"]
    depth_even, depth_even_err = stats["depth_even"]
    odd_even_sigma = np.nan
    if np.isfinite(depth_odd_err) and np.isfinite(depth_even_err):
        combined_err = np.sqrt(depth_odd_err**2 + depth_even_err**2)
        if combined_err > 0:
            odd_even_sigma = abs(depth_odd - depth_even) / combined_err
    if np.isfinite(odd_even_sigma) and odd_even_sigma > 3:
        flags.append(f"odd/even mismatch at {odd_even_sigma:.1f} sigma")
    if best["depth"] * 1e6 > depth_ppm_eb_threshold:
        flags.append(f"depth {best['depth']*1e6:,.0f} ppm too deep for planet")
    if best["duration"] / best["period"] > duration_period_ratio_threshold:
        flags.append("transit duration too long relative to period")
    if best["depth_snr"] < snr_threshold:
        flags.append(f"SNR {best['depth_snr']:.1f} below threshold {snr_threshold}")
    return len(flags) > 0, flags, odd_even_sigma


def search_target(lc_flat, target_name, sector, min_period=0.5, max_period=15.0,
                  n_bootstrap=200, snr_threshold=7.0, random_state=42, verbose=True):
    time, flux, flux_err = _extract_arrays(lc_flat)
    if verbose:
        print(f"[{target_name}] {len(time)} cadences, "
              f"{time.max()-time.min():.2f} days")
    bls, result, best = run_bls(time, flux, flux_err,
                                min_period=min_period, max_period=max_period)
    stats = bls.compute_stats(best["period"], best["duration"], best["t0"])
    period_err_peak = period_uncertainty_from_peak(result)
    if verbose:
        print(f"[{target_name}] P={best['period']:.5f}d  "
              f"depth={best['depth']*1e6:.0f}ppm  SNR={best['depth_snr']:.1f}")
        print(f"[{target_name}] running bootstrap...")
    boot = bootstrap_errors(time, flux, flux_err, best,
                            n_boot=n_bootstrap, random_state=random_state)
    is_fp, fp_reasons, odd_even_sigma = vet_candidate(bls, best, stats,
                                                      snr_threshold=snr_threshold)
    duration_hr = best["duration"] * 24.0
    depth_ppm = best["depth"] * 1e6
    summary_table = pd.DataFrame([{
        "Target": target_name,
        "Sector": sector,
        "Period (d)": round(best["period"], 5),
        "Period Err (d)": round(boot["period_err_bootstrap"], 5)
                         if np.isfinite(boot["period_err_bootstrap"]) else np.nan,
        "Depth (ppm)": round(depth_ppm, 1),
        "Duration (hr)": round(duration_hr, 3),
        "SNR": round(best["depth_snr"], 2),
        "False Positive": is_fp,
    }])
    return {
        "target_name": target_name,
        "sector": sector,
        "time": time, "flux": flux, "flux_err": flux_err,
        "bls": bls, "bls_result": result, "best": best, "stats": stats,
        "bootstrap": boot,
        "is_false_positive": is_fp,
        "false_positive_reasons": fp_reasons,
        "odd_even_sigma": odd_even_sigma,
        "summary_table": summary_table,
    }


def plot_candidate(results, n_bins=60, figsize=(10, 8), save_path=None):
    time = results["time"]
    flux = results["flux"]
    best = results["best"]
    target_name = results["target_name"]
    sector = results["sector"]
    period = best["period"]
    t0 = best["t0"]
    duration = best["duration"]
    depth = best["depth"]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[1, 1.3], hspace=0.32)
    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(time, flux, s=3, color=DATA_COLOR, alpha=0.5, linewidths=0)
    ax0.axhline(1.0, color="#999999", lw=1, ls="--", zorder=1)
    first_epoch = t0 - period * int(np.ceil((t0 - time.min()) / period))
    epoch = first_epoch
    while epoch <= time.max():
        if epoch >= time.min():
            ax0.axvline(epoch, color=MODEL_COLOR, lw=1.2, alpha=0.6, zorder=0)
        epoch += period
    ax0.set_xlabel("Time (BTJD, days)")
    ax0.set_ylabel("Normalized Flux")
    ax0.set_title(f"{target_name} - Sector {sector}: Flattened Light Curve")

    ax1 = fig.add_subplot(gs[1])
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    phase_sorted, flux_sorted = phase[order], flux[order]
    ax1.scatter(phase_sorted * 24, flux_sorted, s=4, color=DATA_COLOR,
                alpha=0.35, linewidths=0, label="Data (unbinned)")
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (phase_sorted >= bin_edges[i]) & (phase_sorted < bin_edges[i + 1])
        if m.sum() > 0:
            bin_means[i] = np.nanmean(flux_sorted[m])
    ax1.scatter(bin_centers * 24, bin_means, s=22, color=BIN_COLOR,
                zorder=5, label="Binned")
    half_dur_days = duration / 2.0
    model_phase_hr = np.linspace(-0.5, 0.5, 2000) * period * 24
    model_flux = np.ones_like(model_phase_hr)
    model_flux[np.abs(model_phase_hr) <= (half_dur_days * 24)] -= depth
    ax1.plot(model_phase_hr, model_flux, color=MODEL_COLOR, lw=2,
             label="BLS box model", zorder=6)
    window_hr = max(duration * 24 * 4, 2.0)
    ax1.set_xlim(-window_hr, window_hr)
    ax1.set_xlabel("Time from mid-transit (hours)")
    ax1.set_ylabel("Normalized Flux")
    ax1.set_title(f"Phase-folded at P = {period:.5f} d   |   "
                  f"Depth = {depth*1e6:.0f} ppm   |   SNR = {best['depth_snr']:.1f}")
    ax1.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"BLS Transit Candidate: {target_name}", fontsize=15, y=0.99)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ── BATCH RUNNER ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    targets = [
        "L 98-59", "TOI-700", "TOI-270", "GJ 357", "HD 21749",
        "TOI-125", "TOI-396", "TOI-451", "TOI-561", "TOI-620",
        "LTT 1445", "TOI-776", "TOI-824", "TOI-1235", "TOI-1634",
        "TOI-1685", "TOI-2136", "TOI-2285", "GJ 3473", "TOI-178"
    ]

    os.makedirs("candidates", exist_ok=True)
    all_tables = []

    for target in targets:
        try:
            print(f"\n{'='*50}")
            print(f"Processing: {target}")
            search = lk.search_lightcurve(target, sector=2, author="SPOC")
            if len(search) == 0:
                print(f"  No SPOC data in sector 2, skipping")
                continue
            lc = search[0].download()
            lc_flat = lc.flatten()
            results = search_target(lc_flat, target_name=target, sector=2, n_bootstrap=50)
            save_path = f"candidates/{target.replace(' ', '_')}_sector2.png"
            fig = plot_candidate(results, save_path=save_path)
            plt.close(fig)
            all_tables.append(results["summary_table"])
            print(f"  Saved: {save_path}")
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    if all_tables:
        final_table = pd.concat(all_tables, ignore_index=True)
        final_table.to_csv("candidates/all_candidates.csv", index=False)
        print(f"\n{'='*50}")
        print(f"COMPLETE: {len(all_tables)} targets processed")
        print(f"\nFull candidate table:")
        print(final_table.to_string(index=False))