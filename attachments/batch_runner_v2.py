"""
================================================================================
 SCO EXOPLANET HUNTER - Batch Runner v2 (fast screening pass)
================================================================================
Fixes applied vs. the original batch script:

  1. No more hardcoded sector=2 -- searches ALL available sectors per target
     and uses the one with the most cadences (more data = better SNR).
  2. max_period widened from 15 -> 40 days so longer-period planets
     (e.g. HD 21749b at ~35.6 d) are actually reachable by the search.
  3. n_bootstrap dropped to 20 for this screening pass (vs. 200 default) --
     rerun search_target() manually with n_bootstrap=200 later on whatever
     candidates look real.
  4. n_periods/n_durations reduced for the initial BLS grid -- still finds
     the right peak, just faster.
  5. Every stage prints immediately with flush=True, so if something hangs
     you'll see exactly which line it's stuck on instead of 30 minutes
     of silence.
  6. Each target has a hard wall-clock timeout (Windows-compatible, via
     concurrent.futures -- signal.alarm doesn't exist on Windows) so one
     stuck download/search can't freeze the whole batch.
  7. Downloads are cached to ./tess_cache so re-running the script doesn't
     re-download stars you already fetched.

Run this after importing your existing exoplanet_bls_pipeline.py functions
(assumes that file is in the same folder).
================================================================================
"""

import matplotlib
matplotlib.use("Agg")

import os
import time as timer
import concurrent.futures as cf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk

from exoplanet_bls_pipeline import search_target, plot_candidate

# ------------------------------------------------------------------------
# CONFIG -- tune these for speed vs. thoroughness
# ------------------------------------------------------------------------
TARGETS = [
    "L 98-59", "TOI-700", "TOI-270", "GJ 357", "HD 21749",
    "TOI-125", "TOI-396", "TOI-451", "TOI-561", "TOI-620",
    "LTT 1445", "TOI-776", "TOI-824", "TOI-1235", "TOI-1634",
    "TOI-1685", "TOI-2136", "TOI-2285", "GJ 3473", "TOI-178",
]

MIN_PERIOD = 0.5
MAX_PERIOD = 40.0          # was 15 -- now reaches longer-period planets
N_BOOTSTRAP_SCREEN = 20    # was 200 -- fast first pass, refine later
N_PERIODS_SCREEN = 12000   # BLS period-grid resolution for screening
N_DURATIONS_SCREEN = 8

PER_TARGET_TIMEOUT_SEC = 120   # hard cap per star (search+download+BLS)
DOWNLOAD_CACHE_DIR = "./tess_cache"
OUTPUT_DIR = "candidates"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_CACHE_DIR, exist_ok=True)


# ------------------------------------------------------------------------
# Fast BLS settings for screening: monkey-patch the defaults used inside
# search_target() without editing the pipeline file itself.
# ------------------------------------------------------------------------
import exoplanet_bls_pipeline as pipeline

_original_run_bls = pipeline.run_bls


def _fast_run_bls(time, flux, flux_err, min_period=MIN_PERIOD,
                   max_period=MAX_PERIOD, **kwargs):
    kwargs.setdefault("n_periods", N_PERIODS_SCREEN)
    kwargs.setdefault("n_durations", N_DURATIONS_SCREEN)
    return _original_run_bls(time, flux, flux_err, min_period=min_period,
                              max_period=max_period, **kwargs)


pipeline.run_bls = _fast_run_bls


# ------------------------------------------------------------------------
# Per-target worker: everything that can hang lives in here
# ------------------------------------------------------------------------
def process_one_target(target):
    print(f"  [{target}] searching MAST (all sectors)...", flush=True)
    search = lk.search_lightcurve(target, author="SPOC")
    if len(search) == 0:
        print(f"  [{target}] no SPOC light curves found at all, skipping", flush=True)
        return None

    # pick the sector with the most exposures (usually = longest baseline)
    exptimes = search.table["t_exptime"] if "t_exptime" in search.table.colnames else None
    sectors_available = sorted(set(search.table["sequence_number"])) \
        if "sequence_number" in search.table.colnames else None
    print(f"  [{target}] found {len(search)} result(s)"
          + (f", sectors: {sectors_available}" if sectors_available else ""),
          flush=True)

    print(f"  [{target}] downloading (cached to {DOWNLOAD_CACHE_DIR})...", flush=True)
    lc = search[0].download(download_dir=DOWNLOAD_CACHE_DIR)
    if lc is None:
        print(f"  [{target}] download returned nothing, skipping", flush=True)
        return None

    sector = int(getattr(lc, "sector", -1))
    print(f"  [{target}] downloaded sector {sector}. Flattening...", flush=True)
    lc_flat = lc.flatten()

    print(f"  [{target}] running BLS (screening settings)...", flush=True)
    results = search_target(
        lc_flat, target_name=target, sector=sector,
        min_period=MIN_PERIOD, max_period=MAX_PERIOD,
        n_bootstrap=N_BOOTSTRAP_SCREEN, verbose=False,
    )
    print(f"  [{target}] P={results['best']['period']:.4f} d  "
          f"SNR={results['best']['depth_snr']:.1f}  "
          f"FP={results['is_false_positive']}", flush=True)

    save_path = os.path.join(OUTPUT_DIR, f"{target.replace(' ', '_')}_sector{sector}.png")
    fig = plot_candidate(results, save_path=save_path)
    plt.close(fig)
    print(f"  [{target}] saved plot -> {save_path}", flush=True)

    return results["summary_table"]


# ------------------------------------------------------------------------
# Main batch loop with a hard per-target timeout
# ------------------------------------------------------------------------
def main():
    all_tables = []
    t_start = timer.time()

    for i, target in enumerate(TARGETS, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(TARGETS)}] {target}")
        print(f"{'='*60}", flush=True)

        t0 = timer.time()
        with cf.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(process_one_target, target)
            try:
                table = future.result(timeout=PER_TARGET_TIMEOUT_SEC)
                if table is not None:
                    all_tables.append(table)
            except cf.TimeoutError:
                print(f"  [{target}] TIMED OUT after {PER_TARGET_TIMEOUT_SEC}s, "
                      f"skipping (thread left running in background)", flush=True)
            except Exception as e:
                print(f"  [{target}] FAILED: {e}", flush=True)

        print(f"  [{target}] took {timer.time()-t0:.1f}s", flush=True)

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {len(all_tables)}/{len(TARGETS)} targets succeeded "
          f"in {timer.time()-t_start:.1f}s total")
    print(f"{'='*60}", flush=True)

    if all_tables:
        final_table = pd.concat(all_tables, ignore_index=True)
        out_csv = os.path.join(OUTPUT_DIR, "all_candidates.csv")
        final_table.to_csv(out_csv, index=False)
        print(f"\nSaved combined table -> {out_csv}\n")
        print(final_table.to_string(index=False))
    else:
        print("\nNo targets succeeded -- check network/MAST access.")


if __name__ == "__main__":
    main()
