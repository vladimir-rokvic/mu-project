import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

PER_FRAME_DIR = "./per_frame_features/"
dct_cols = [f"dct_{i}" for i in range(128)]

all_energy = []

files = sorted(os.listdir(PER_FRAME_DIR))[:50]   # sample 50 videos

for fname in files:
    df = pd.read_csv(os.path.join(PER_FRAME_DIR, fname)).dropna()

    if len(df) == 0 or not all(c in df.columns for c in dct_cols):
        continue

    # For each frame compute energy distribution
    for _, row in df.iterrows():
        coeffs = row[dct_cols].values.astype(float)
        energy = coeffs ** 2
        total  = energy.sum()

        if total == 0:
            continue

        cum_energy = np.cumsum(energy) / total
        all_energy.append(cum_energy)

all_energy = np.array(all_energy)
mean_energy = all_energy.mean(axis=0)

# Print thresholds
for threshold in [0.80, 0.90, 0.95, 0.99]:
    hits = np.where(mean_energy >= threshold)[0]
    if len(hits) > 0:
        print(f"{threshold*100:.0f}% energy → {hits[0]+1} coefficients")
    else:
        print(f"{threshold*100:.0f}% energy → not reached in 128 coefficients")

# Plot
plt.figure(figsize=(10, 4))
plt.plot(mean_energy, label='mean cumulative energy')
plt.fill_between(range(128),
                 all_energy.min(axis=0),
                 all_energy.max(axis=0),
                 alpha=0.2, label='min-max range')
plt.axhline(0.90, color='r', linestyle='--', label='90%')
plt.axhline(0.95, color='g', linestyle='--', label='95%')
plt.axhline(0.99, color='b', linestyle='--', label='99%')
plt.xlabel("DCT coefficient index")
plt.ylabel("Cumulative energy")
plt.title("DCT energy across extracted lip frames")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
