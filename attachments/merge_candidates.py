import pandas as pd
import os

BASE = r"C:\Users\Мой Компьютер\OneDrive\Desktop\exoplanet_projects\candidates"
df1 = pd.read_csv(os.path.join(BASE, "all_candidates.csv"))
df2 = pd.read_csv(os.path.join(BASE, "new_candidates_batch2.csv"))

df_all = pd.concat([df1, df2], ignore_index=True)
df_all.to_csv(os.path.join(BASE, "master_candidates.csv"), index=False)

print(f"Total candidates: {len(df_all)}")
print(f"Strong (SNR>10, not FP): {len(df_all[(df_all['SNR']>10) & (~df_all['False Positive'])])}")
print(f"Candidate (SNR 7-10, not FP): {len(df_all[(df_all['SNR']>=7) & (df_all['SNR']<=10) & (~df_all['False Positive'])])}")
print(f"Flagged: {len(df_all[df_all['False Positive']])}")
print()

print(
    df_all[
        ["Target","Sector","Period (d)","SNR","Depth (ppm)","False Positive"]
    ].sort_values("SNR", ascending=False).to_string(index=False)
)