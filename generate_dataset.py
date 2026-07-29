"""
Dataset Generator — Crop Recommendation
Generates a 2200-record dataset matching the real Kaggle Crop Recommendation
dataset's statistical distributions (N, P, K, temperature, humidity, ph, rainfall).
Each of the 22 crops has 100 records with realistic feature ranges from agronomy literature.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Each crop: [N_mean, N_std, P_mean, P_std, K_mean, K_std,
#             temp_mean, temp_std, humidity_mean, humidity_std,
#             ph_mean, ph_std, rain_mean, rain_std]
CROP_PARAMS = {
    "rice":        [80,  6,  40,  4,  40,  4,  23.5, 1.5, 82,  4,  6.4, 0.3, 236, 25],
    "maize":       [78,  8,  48,  5,  20,  4,  22.5, 2.0, 65,  7,  6.2, 0.4, 67,  15],
    "chickpea":    [40,  5,  67,  5,  79,  5,  18.8, 1.5, 17,  4,  7.0, 0.3, 81,  12],
    "kidneybeans": [21,  4,  67,  5,  19,  4,  19.8, 1.8, 22,  5,  5.7, 0.4, 105, 15],
    "pigeonpeas":  [21,  4,  67,  5,  19,  4,  27.5, 1.5, 49,  5,  5.8, 0.4, 149, 20],
    "mothbeans":   [21,  4,  48,  5,  19,  4,  28.2, 1.5, 53,  6,  6.9, 0.4, 52,  10],
    "mungbean":    [21,  4,  47,  5,  19,  4,  29.0, 1.5, 89,  4,  6.7, 0.3, 50,  10],
    "blackgram":   [40,  5,  67,  5,  19,  4,  30.0, 1.5, 66,  5,  7.0, 0.4, 68,  12],
    "lentil":      [19,  4,  68,  5,  19,  4,  24.5, 1.5, 65,  5,  6.9, 0.3, 46,  10],
    "pomegranate": [18,  4,  18,  4,  40,  4,  21.9, 1.5, 90,  4,  6.5, 0.3, 107, 15],
    "banana":      [100, 8,  82,  6,  50,  5,  27.4, 1.5, 80,  5,  5.8, 0.4, 105, 15],
    "mango":       [20,  4,  27,  4,  29,  4,  31.2, 1.5, 50,  6,  5.8, 0.4, 95,  15],
    "grapes":      [23,  4,  132, 8,  200, 10, 23.8, 1.5, 81,  5,  6.0, 0.3, 70,  12],
    "watermelon":  [99,  7,  17,  4,  50,  5,  25.6, 1.5, 85,  5,  6.5, 0.3, 51,  10],
    "muskmelon":   [100, 7,  17,  4,  50,  5,  28.7, 1.5, 92,  4,  6.4, 0.3, 25,  8],
    "apple":       [21,  4,  134, 8,  200, 10, 22.6, 1.5, 92,  4,  5.9, 0.3, 113, 15],
    "orange":      [20,  4,  16,  4,  10,  3,  11.0, 1.5, 92,  4,  7.0, 0.3, 114, 15],
    "papaya":      [49,  5,  59,  5,  50,  5,  33.7, 1.5, 92,  4,  6.7, 0.3, 143, 20],
    "coconut":     [22,  4,  16,  4,  30,  4,  27.1, 1.5, 95,  3,  5.9, 0.3, 175, 20],
    "cotton":      [118, 8,  46,  5,  19,  4,  24.0, 1.5, 79,  5,  6.9, 0.3, 82,  15],
    "jute":        [78,  7,  46,  5,  39,  4,  24.9, 1.5, 80,  5,  6.7, 0.3, 175, 20],
    "coffee":      [101, 7,  28,  4,  29,  4,  25.5, 1.5, 58,  6,  6.8, 0.3, 159, 20],
}

records = []
for crop, params in CROP_PARAMS.items():
    n_m,  n_s  = params[0],  params[1]
    p_m,  p_s  = params[2],  params[3]
    k_m,  k_s  = params[4],  params[5]
    t_m,  t_s  = params[6],  params[7]
    h_m,  h_s  = params[8],  params[9]
    ph_m, ph_s = params[10], params[11]
    r_m,  r_s  = params[12], params[13]

    for _ in range(100):
        records.append({
            "N":           max(0, round(np.random.normal(n_m,  n_s),  2)),
            "P":           max(0, round(np.random.normal(p_m,  p_s),  2)),
            "K":           max(0, round(np.random.normal(k_m,  k_s),  2)),
            "temperature": round(np.random.normal(t_m,  t_s),  2),
            "humidity":    round(np.clip(np.random.normal(h_m, h_s), 1, 100), 2),
            "ph":          round(np.clip(np.random.normal(ph_m, ph_s), 3.5, 9.5), 2),
            "rainfall":    max(0, round(np.random.normal(r_m,  r_s),  2)),
            "label":       crop,
        })

df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
out_path = os.path.join(os.path.dirname(__file__), "data", "crop_recommendation.csv")
df.to_csv(out_path, index=False)
print(f"[OK] Dataset saved: {out_path}")
print(f"     Shape : {df.shape}")
print(f"     Crops : {df['label'].nunique()} ({', '.join(sorted(df['label'].unique()))})")
print(f"\nSample statistics:")
print(df.describe().round(2))
