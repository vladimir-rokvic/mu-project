import os
import pandas as pd
import numpy as np

geometric_features = [
        "width",
        "height",
        "hull_area",
        "aspect_ratio",
        "angle",
        "ellipse_area",
        "solidity",
        "upper_lip_height",
        "lower_lip_height",
        "circularity"
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

N_TIMESTEPS = 10

test_speakers = [4, 8, 14, 26, 28]

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

def temporal_agregate(df, feature):
    name_mean = "d_" + feature + "_mean"
    name_std =  "d_" + feature + "_std"
    name_max =  "d_" + feature + "_max"
    name_min =  "d_" + feature + "_min"

    ret = {
            name_mean: df[feature].mean(),
            name_std: df[feature].std(),
            name_max: df[feature].max(),
            name_min: df[feature].min()
    }
    
    return ret

def interpolate(df, feature):
    f_old = df[feature]
    x_old = np.linspace(0, 1, len(df))
    x_new = np.linspace(0, 1, N_TIMESTEPS)

    f_new = np.interp(x_new, x_old, f_old)

    ret = {}
    i = 0
    for f_val in f_new:
        name = feature + "_t" + str(i)
        ret[name] = f_val

        i += 1

    return ret

def timed(df, feature):
    ret = {}

    name_max = feature + "_tmax"
    name_min = feature + "_tmin"

    ret[name_max] = df[feature].idxmax() / len(df)
    ret[name_min] = df[feature].idxmin() / len(df)

    return ret

def temporal_timed(df, feature):
    ret = {}

    name_max = feature + "_tmax"
    name_min = feature + "_tmin"

    ret[name_max] = df[feature].idxmax() / len(df)
    ret[name_min] = df[feature].idxmin() / len(df)

    return ret

def agregate_features(csv_path: str, label: int, spk: int):
    df = pd.read_csv(csv_path)
    d_df = df[geometric_features].diff().fillna(0)
    ret = {}

    for feature in geometric_features:
        ret.update(agregate(df, feature))
        ret.update(temporal_agregate(d_df, feature))
    
    ret["y"] = label
    ret["spk"] = spk

    return ret

def interpolate_features(csv_path: str, label: int, spk: int):
    df = pd.read_csv(csv_path)
    ret = {}

    for feature in geometric_features:
        ret.update(interpolate(df, feature))
    
    ret["y"] = label
    ret["spk"] = spk

    return ret

def time_features(csv_path: str, label: int, spk: int):
    df = pd.read_csv(csv_path)
    d_df = df[geometric_features].diff().fillna(0)
    ret = {}

    for feature in geometric_features:
        ret.update(timed(df, feature))
    
    ret["duration"] = len(df)
    ret["y"] = label
    ret["spk"] = spk

    return ret


PER_FRAME_DIR = "./per_frame_features/"

per_frame_features = sorted(os.listdir(PER_FRAME_DIR))
i = 0
X_agregated = []
X_interp = []
X_timed = []
for frame_features in per_frame_features:
    #print(frame_features)
    csv_path = os.path.join(PER_FRAME_DIR, frame_features)
    X_agregated.append(agregate_features(csv_path, i % 30, i // 30 + 1))
    X_interp.append(interpolate_features(csv_path, i % 30, i // 30 + 1))
    X_timed.append(time_features(csv_path, i % 30, i // 30 + 1))

    i += 1

df = pd.DataFrame(X_agregated)
df.to_csv('geometric_agregated.csv', index=False)

df_interp = pd.DataFrame(X_interp)
df_interp.to_csv('geometric_interp.csv', index=False)

df_timed = pd.DataFrame(X_timed)
df_timed.to_csv('geometric_timed.csv', index=False)
