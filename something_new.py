import os
import numpy as np
import pandas as pd
import cv2 as cv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

VIDEO_DIR = "./iseckane_reci_masked/videos/"
ROI_DIR   = "./iseckane_reci_masked/roi/"
PER_FRAME_DIR = "./per_frame_features/"

CROP_SIZE    = (64, 32)    # W x H — keeps lip aspect ratio
N_COMPONENTS = 20          # eigenlips to keep
N_TIMESTEPS  = 20
test_speakers = [4, 8, 14, 26, 28]

# ── 1. Collect all training frames as pixel vectors ─────────────

def load_frame_pixels(video_path, roi_path):
    """Load all frames from one video as flattened grayscale crops."""
    video = cv.VideoCapture(video_path)
    roi   = pd.read_csv(roi_path)
    idx   = 0
    frames = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        row = roi.iloc[idx]
        x, y, w, h = int(row['x']), int(row['y']), \
                     int(row['width']), int(row['height'])
        crop = frame[y:y+h, x:x+w]

        if crop.size == 0:
            idx += 1
            continue

        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, CROP_SIZE).astype(float)
        gray /= 255.0                           # normalize to [0,1]

        frames.append(gray.flatten())           # (2048,)
        idx += 1

    video.release()
    return np.array(frames)                     # (T, 2048)


print("Collecting training frames for eigenlip PCA...")
videos = sorted(os.listdir(VIDEO_DIR))
train_frames_all = []

for i, vname in enumerate(videos):
    spk = i // 30 + 1
    if spk in test_speakers:
        continue                                # skip test speakers

    vpath = os.path.join(VIDEO_DIR, vname)
    rpath = os.path.join(ROI_DIR, vname.replace(".mp4", ".txt"))
    frames = load_frame_pixels(vpath, rpath)

    if len(frames) > 0:
        train_frames_all.append(frames)

    print(f"  [{i+1}/900] {vname}", end="\r")

train_frames_all = np.vstack(train_frames_all)  # (N_train_frames, 2048)
print(f"\nTotal training frames: {train_frames_all.shape}")


# ── 2. Fit eigenlip PCA on training frames only ─────────────────

print(f"Fitting PCA with {N_COMPONENTS} components...")
sc_pixel  = StandardScaler()
pca_pixel = PCA(n_components=N_COMPONENTS, whiten=False)

train_sc  = sc_pixel.fit_transform(train_frames_all)
pca_pixel.fit(train_sc)

var_explained = pca_pixel.explained_variance_ratio_.sum() * 100
print(f"Variance explained by {N_COMPONENTS} components: {var_explained:.1f}%")

# Visualize eigenlips (optional)
def show_eigenlips(pca, crop_size, n=10):
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
    for i in range(n):
        eigenlip = pca.components_[i].reshape(crop_size[1], crop_size[0])
        eigenlip = (eigenlip - eigenlip.min()) / \
                   (eigenlip.max() - eigenlip.min() + 1e-6)
        axes[i].imshow(eigenlip, cmap='gray')
        axes[i].set_title(f"PC{i+1}")
        axes[i].axis('off')
    plt.suptitle("Eigenlips — top PCA components of lip appearance")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
show_eigenlips(pca_pixel, CROP_SIZE, n=min(10, N_COMPONENTS))


# ── 3. Extract eigenlip features for every video ─────────────────

def extract_eigenlip_features(video_path, roi_path, label, spk):
    frames = load_frame_pixels(video_path, roi_path)

    if len(frames) == 0:
        return None

    # Project to eigenlip space
    frames_sc  = sc_pixel.transform(frames)          # (T, 2048)
    frames_pca = pca_pixel.transform(frames_sc)       # (T, N_COMPONENTS)

    T   = len(frames_pca)
    ret = {}

    for comp in range(N_COMPONENTS):
        series = frames_pca[:, comp]                  # (T,)

        # Interpolate to fixed length
        x_orig = np.linspace(0, 1, T)
        x_new  = np.linspace(0, 1, N_TIMESTEPS)
        interp = np.interp(x_new, x_orig, series)

        for t, val in enumerate(interp):
            ret[f"elip_pc{comp}_t{t}"] = val

    ret["y"] = label
    ret["spk"] = spk
    return ret


print("\nExtracting eigenlip features for all videos...")
rows = []

for i, vname in enumerate(videos):
    vpath = os.path.join(VIDEO_DIR, vname)
    rpath = os.path.join(ROI_DIR, vname.replace(".mp4", ".txt"))

    row = extract_eigenlip_features(vpath, rpath, i % 30, i // 30 + 1)
    if row is not None:
        rows.append(row)

    print(f"[{i+1}/900] {vname}", end="\r")

df = pd.DataFrame(rows)
df.to_csv("eigenlip_features.csv", index=False)
print(f"\nSaved eigenlip_features.csv  shape={df.shape}")


# ── 4. Train SVM ────────────────────────────────────────────────

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

feat_cols = [c for c in df.columns if c not in ["y", "spk"]]
X_train = df.loc[~df["spk"].isin(test_speakers), feat_cols].values
y_train = df.loc[~df["spk"].isin(test_speakers), "y"].values
X_test  = df.loc[df["spk"].isin(test_speakers),  feat_cols].values
y_test  = df.loc[df["spk"].isin(test_speakers),  "y"].values

print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, C, kernel in [
    ("rbf C=1",   1,   "rbf"),
    ("rbf C=10",  10,  "rbf"),
    ("lin C=1",   1,   "linear"),
    ("lin C=10",  10,  "linear"),
]:
    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("svm", SVC(kernel=kernel, C=C, gamma="scale"))
    ])
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"{name:12s}  CV: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

# Best config test score
pipe = Pipeline([
    ("sc",  StandardScaler()),
    ("svm", SVC(kernel="linear", C=10, gamma="scale"))
])
pipe.fit(X_train, y_train)
print(f"\nTest accuracy:  {pipe.score(X_test, y_test)*100:.1f}%")
print(f"Train accuracy: {pipe.score(X_train, y_train)*100:.1f}%")
