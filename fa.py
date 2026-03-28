import os
import pandas as pd
import numpy as np

geometric_features = [
        "height",
        "area",
        "aspect_ratio",
        "angle",
        "ellipse_ratio",
        "corner_height_diff",
        "upper_lip_height",
        "lower_lip_height"
        ]

d_geometric_features = [
        "d_height",
        "d_area",
        "d_aspect_ratio",
        "d_angle",
        "d_ellipse_ratio",
        "d_corner_height_diff",
        "d_upper_lip_height",
        "d_lower_lip_height"
        ]

test_speakers = [4, 8, 14, 26, 28]

def z_norm(X, features):
    df = X.copy()
    train_mask = ~df["spk"].isin(test_speakers)

    for spk in df["spk"].unique():
        mask = df["spk"] == spk

        if spk in test_speakers:
            mean = df.loc[train_mask, features].mean()
            std = df.loc[train_mask, features].std() + 1e-6
        else:
            mean = df.loc[mask, features].mean()
            std = df.loc[mask, features].std() + 1e-6

        df.loc[mask, features] = (df.loc[mask, features] - mean) / std

    return df

def agregate(df, feature):
    name_mean = feature + "_mean"
    name_std = feature + "_std"
    name_max = feature + "_max"
    name_min = feature + "_min"

    ret = {
            name_mean: df[feature].mean(),
            name_std: df[feature].std(),
            name_max: df[feature].max(),
            name_min: df[feature].min()
    }
    
    return ret

def agregate_features(csv_path: str, label: int, spk: int):
    df = pd.read_csv(csv_path)
    ret = {}

    for feature in geometric_features:
        ret.update(agregate(df, feature))
    
    ret["y"] = label
    ret["spk"] = spk

    return ret

PER_FRAME_DIR = "./per_frame_features/"

per_frame_features = sorted(os.listdir(PER_FRAME_DIR))
i = 0
X = []
for frame_features in per_frame_features:
    #print(frame_features)
    csv_path = os.path.join(PER_FRAME_DIR, frame_features)
    X.append(agregate_features(csv_path, i % 30, i // 30 + 1))

    i += 1
df = pd.DataFrame(X)

features = [c for c in df.columns if c not in ["spk", "y"]]
df = z_norm(df, features)

df.to_csv('agregated_features_2.csv', index=False)
