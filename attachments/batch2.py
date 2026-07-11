import lightkurve as lk
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Add your pipeline path
sys.path.append(r"C:\Users\Мой Компьютер\OneDrive\Aisha - Personal\Desktop\exoplanet_projects")
from exoplanet_bls_pipeline import search_target, plot_candidate

BASE_PATH = r"C:\Users\Мой Компьютер\OneDrive\Aisha - Personal\Desktop\exoplanet_projects\candidates"
CSV_PATH = os.path.join(BASE_PATH, "all_candidates.csv")

# ── 30 NEW UNIQUE CONFIRMED TESS PLANET HOSTS ────────────────────────────────
# Grouped by type so you understand what you're running:
# USP = ultra-short-period (P < 1 day) → highest SNR, easiest detection
# MULTI = confirmed multi-planet system → scientifically rich
# HOT = hot Jupiter/Neptune → very deep transits
# NOTABLE = scientifically significant for specific reason

targets_new = [
    # USP planets - expect very high SNR
    "TOI-431",       # USP P=0.49d confirmed + outer planet P=12.5d
    "TOI-500",       # USP P=0.55cleard confirmed
    "TOI-1075",      # USP P=0.61d confirmed super-Earth
    "TOI-1516",      # P=1.03d confirmed hot Saturn
    "LTT 3780",      # P=0.77d + P=12.25d confirmed multi

    # Multi-planet systems - scientifically rich
    "GJ 9827",       # 3 confirmed planets P=1.21, 3.65, 6.20d
    "HD 63433",      # 2 confirmed planets P=7.11, 20.55d
    "HD 73583",      # 2 confirmed planets P=6.40, 18.56d
    "TOI-836",       # 2 confirmed planets P=3.82, 8.60d
    "TOI-1130",      # 2 confirmed planets P=4.07, 8.35d
    "TOI-1233",      # HD 108236: 5 confirmed planets (3.8 to 29.5d)
    "HD 110067",     # 6-planet resonant chain, landmark system
    "TOI-421",       # 2 confirmed planets P=5.20, 16.07d
    "TOI-402",       # 2 confirmed planets P=4.76, 17.18d

    # Short-medium period confirmed detections
    "TOI-519",       # P=1.27d confirmed
    "TOI-544",       # P=1.55d confirmed
    "TOI-674",       # P=1.98d confirmed sub-Neptune
    "TOI-628",       # P=3.41d confirmed hot Jupiter
    "TOI-763",       # P=5.91d confirmed
    "TOI-1064",      # P=6.44d confirmed

    # Hot Jupiters / deep transits - easy to detect
    "TOI-1431",      # P=2.65d confirmed hot Jupiter
    "TOI-1518",      # P=1.90d confirmed hot Jupiter
    "TOI-1670",      # P=10.9d confirmed sub-Saturn

    # Notable/scientifically interesting
    "TOI-1452",      # P=11.1d confirmed ocean world candidate
    "TOI-1266",      # 2 confirmed planets P=10.9, 18.8d
    "TOI-216",       # 2 confirmed resonant planets P=17.2, 34.5d
    "TOI-132",       # P=19.0d confirmed Neptune
    "TOI-892",       # P=10.63d confirmed
    "LHS 1478",      # P=1.95d confirmed M-dwarf hosted
    "HD 191939",     # 6 confirmed planets spanning 8.9 to 284d
]

print(f"Starting batch run: {len(targets_new)} new targets")
print("="*60)

os.makedirs(BASE_PATH, exist_ok=True)
new_tables = []
failed = []

for i, target in enumerate(targets_new, start=1):
    print(f"\n[{i}/{len(targets_new)}] Processing: {target}")
    try:
        # Search all available SPOC data, pick best sector automatically
        search = lk.search_lightcurve(target, author="SPOC", exptime=120)

        if len(search) == 0:
            # Try TESS-SPOC 
            search = lk.search_lightcurve(target, author="TESS-SPOC")

        if len(search) == 0:
            print(f"  No SPOC data found for {target}, skipping")
            failed.append(target)
            continue

        # Pick the sector with most data points (usually best quality)
        lc_raw = search[0].download()
        sector = int(lc_raw.meta.get("SECTOR", 0))
        lc_flat = lc_raw.remove_nans().remove_outliers(sigma=5).flatten()

        results = search_target(
            lc_flat,
            target_name=target,
            sector=sector,
            n_bootstrap=150,
            verbose=True
        )

        safe_name = target.replace(" ", "_").replace("/", "_")
        save_path = os.path.join(BASE_PATH, f"{safe_name}_sector{sector}.png")

        fig = plot_candidate(results, save_path=save_path)
        plt.close(fig)

        new_tables.append(results["summary_table"])
        snr = results["best"]["depth_snr"]
        period = results["best"]["period"]
        fp = results["is_false_positive"]
        print(f"  DONE | Period={period:.4f}d | SNR={snr:.1f} | FP={fp} | Saved: {save_path}")

    except Exception as e:
        print(f"  FAILED: {e}")
        failed.append(target)
        continue

# Save combined results
print("\n" + "="*60)
print(f"Completed: {len(new_tables)} succeeded, {len(failed)} failed")

if new_tables:
    new_df = pd.concat(new_tables, ignore_index=True)
    new_csv = os.path.join(BASE_PATH, "new_candidates_batch2.csv")
    new_df.to_csv(new_csv, index=False)
    print(f"\nBatch 2 results saved to: {new_csv}")
    print("\nFull results:")
    print(new_df[["Target", "Sector", "Period (d)", "SNR", "Depth (ppm)", "False Positive"]].to_string(index=False))

if failed:
    print(f"\nFailed targets (no SPOC data or error): {failed}")
    print("These can be manually replaced with alternative targets")