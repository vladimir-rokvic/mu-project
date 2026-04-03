import os
import numpy as np
import pandas as pd
import cv2 as cv
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

VIDEO_DIR  = "./iseckane_reci_masked/videos/"
ROI_DIR    = "./iseckane_reci_masked/roi/"
CROP_SIZE  = (64, 32)      # W × H — keeps lip aspect ratio
N_TIMESTEPS  = 20
N_COMPONENTS = 20          # PCA components for HOG
test_speakers = [4, 8, 14, 26, 28]


# ── 1. HOG for one frame ────────────────────────────────────────

def get_hog(crop_bgr):
    gray = cv.cvtColor(crop_bgr, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, CROP_SIZE)
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    return feat   # fixed size regardless of crop dimensions


# ── 2. Load HOG frames for one video ───────────────────────────

def load_hog_frames(video_path, roi_path):
    video  = cv.VideoCapture(video_path)
    roi    = pd.read_csv(roi_path)
    frames = []
    idx    = 0

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

        frames.append(get_hog(crop))
        idx += 1

    video.release()
    return np.array(frames)   # (T, hog_dim)


# ── 3. Collect training frames for PCA ─────────────────────────

print("Collecting training HOG frames...")
videos       = sorted(os.listdir(VIDEO_DIR))
train_frames = []

for i, video_name in enumerate(videos):
    if (i // 30 + 1) in test_speakers:
        continue

    vpath  = os.path.join(VIDEO_DIR, video_name)
    rpath  = os.path.join(ROI_DIR, video_name.replace('.mp4', '.txt'))
    frames = load_hog_frames(vpath, rpath)

    if len(frames) > 0:
        train_frames.append(frames)

    print(f"  [{i+1}/900]", end="\r")

train_frames = np.vstack(train_frames)   # (N_train_frames, hog_dim)
print(f"\nTraining frames: {train_frames.shape}")
print(f"HOG dim per frame: {train_frames.shape[1]}")


# ── 4. Fit PCA on training frames ───────────────────────────────

sc  = StandardScaler()
pca = PCA(n_components=N_COMPONENTS, whiten=True, random_state=42)

train_sc = sc.fit_transform(train_frames)
pca.fit(train_sc)
print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")


# ── 5. Build video-level feature vectors ────────────────────────

def get_row(video_path, roi_path, label, spk):
    frames = load_hog_frames(video_path, roi_path)

    if len(frames) == 0:
        return None

    # Project to PCA space
    frames_pca = pca.transform(sc.transform(frames))  # (T, N_COMPONENTS)

    T     = len(frames_pca)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, N_TIMESTEPS)
    ret   = {}

    for i in range(N_COMPONENTS):
        series = frames_pca[:, i]
        interp = np.interp(x_new, x_old, series)

        for t, val in enumerate(interp):
            ret[f"hog_pc{i}_t{t}"] = val

        ret[f"hog_pc{i}_mean"]  = series.mean()
        ret[f"hog_pc{i}_std"]   = series.std()
        ret[f"hog_pc{i}_max"]   = series.max()
        ret[f"hog_pc{i}_min"]   = series.min()
        ret[f"hog_pc{i}_range"] = series.max() - series.min()

    ret["y"]   = label
    ret["spk"] = spk
    return ret


print("\nExtracting features...")
rows = []

for i, video_name in enumerate(videos):
    vpath = os.path.join(VIDEO_DIR, video_name)
    rpath = os.path.join(ROI_DIR, video_name.replace('.mp4', '.txt'))

    row = get_row(vpath, rpath, i % 30, i // 30 + 1)
    if row is not None:
        rows.append(row)

    print(f"[{i+1}/900] {video_name}", end="\r")

df = pd.DataFrame(rows)
df.to_csv("hog_pca_features.csv", index=False)
print(f"\nSaved hog_pca_features.csv  shape={df.shape}")
print(df["y"].value_counts().sort_index())
print(df["spk"].value_counts().sort_index())


# ── 6. SVM Classification ───────────────────────────────────────

feat_cols = [c for c in df.columns if c not in ["y", "spk"]]

X_train = df.loc[~df["spk"].isin(test_speakers), feat_cols].values
y_train = df.loc[~df["spk"].isin(test_speakers), "y"].values
X_test  = df.loc[df["spk"].isin(test_speakers),  feat_cols].values
y_test  = df.loc[df["spk"].isin(test_speakers),  "y"].values

print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Quick baseline
configs = [
    ("rbf  C=1",  Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf",    C=1,   random_state=42))])),
    ("rbf  C=10", Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf",    C=10,  random_state=42))])),
    ("lin  C=1",  Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="linear", C=1,   random_state=42))])),
    ("lin  C=10", Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="linear", C=10,  random_state=42))])),
]

print("\n--- Baseline ---")
for name, pipe in configs:
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=cv, scoring="accuracy", n_jobs=-1)
    pipe.fit(X_train, y_train)
    print(f"{name:12s}  CV: {scores.mean()*100:.1f}%  "
          f"Test: {pipe.score(X_test, y_test)*100:.1f}%")

# Grid search
print("\n--- Grid Search ---")
param_grid = {
    "svm__kernel": ["rbf", "linear"],
    "svm__C":      [0.1, 1, 10, 100],
    "svm__gamma":  ["scale", "auto"],
}

pipe = Pipeline([
    ("sc",  StandardScaler()),
    ("svm", SVC(random_state=42))
])

search = GridSearchCV(pipe, param_grid, cv=cv,
                      scoring="accuracy", n_jobs=-1, verbose=1)
search.fit(X_train, y_train)

print(f"Best params:    {search.best_params_}")
print(f"Best CV score:  {search.best_score_*100:.1f}%")
print(f"Test accuracy:  {search.score(X_test, y_test)*100:.1f}%")
print(f"Train accuracy: {accuracy_score(y_train, search.predict(X_train))*100:.1f}%")
