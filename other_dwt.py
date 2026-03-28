import pywt
import cv2 as cv
import numpy as np
import pandas as pd
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

VIDEO_DIR  = "./iseckane_reci_masked/videos/"
ROI_DIR    = "./iseckane_reci_masked/roi/"

# ── Paper parameters adapted for your setup ────────────────────
ROI_SIZE    = (64, 45)    # paper uses 32×20 — small and focused
WAVELET     = 'db4'       # paper uses db4
LEVEL       = 3           # paper uses 3 levels
N_FRAMES    = 10          # paper uses 10 significant frames
test_speakers = [4, 8, 14, 26, 28]


# ── 1. DWT on one frame → LL approximation ─────────────────────

def frame_to_ll(frame_gray):
    """
    Apply 2D DWT, return only LL (low frequency approximation).
    Paper: 32×20 input → LL3 = 5×6 = 30 coefficients
    """
    coeffs = pywt.wavedec2(frame_gray, WAVELET, level=LEVEL)
    LL     = coeffs[0]   # approximation at deepest level
    return LL.flatten()  # (30,) for 32×20 input at level 3


# ── 2. Select N significant frames per video ───────────────────

def select_significant_frames(frames_gray, n=N_FRAMES):
    """
    Paper selects 10 most significant frames using mean square
    difference between consecutive frames — frames with most
    motion are most informative.
    """
    if len(frames_gray) <= n:
        # Pad with last frame if too short
        while len(frames_gray) < n:
            frames_gray.append(frames_gray[-1])
        return frames_gray

    # Compute frame differences
    diffs = []
    for i in range(len(frames_gray) - 1):
        diff = np.mean((frames_gray[i].astype(float) -
                        frames_gray[i+1].astype(float)) ** 2)
        diffs.append((diff, i))

    # Sort by difference descending — most motion first
    diffs.sort(reverse=True)

    # Take top n frame indices, sort chronologically
    selected_idx = sorted([idx for _, idx in diffs[:n]])
    return [frames_gray[i] for i in selected_idx]


# ── 3. Load video → DWT feature vector ─────────────────────────

def load_dwt_vector(video_path, roi_path):
    """
    Following the paper:
    - Load all frames
    - Select 10 significant frames
    - Apply 2D DWT to each → LL coefficients
    - Concatenate: 10 × 30 = 300-dim vector per word
    """
    video  = cv.VideoCapture(video_path)
    roi    = pd.read_csv(roi_path)
    idx    = 0
    gray_frames = []

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
        gray = cv.resize(gray, ROI_SIZE).astype(float) / 255.0
        gray_frames.append(gray)
        idx += 1

    video.release()

    if len(gray_frames) == 0:
        return None

    # Select significant frames
    selected = select_significant_frames(gray_frames, N_FRAMES)

    # Apply DWT to each selected frame
    dwt_frames = [frame_to_ll(f) for f in selected]

    # Stack → one vector per word (paper: 300-dim)
    return np.concatenate(dwt_frames)   # (N_FRAMES × LL_size,)


# ── 4. Extract all videos ───────────────────────────────────────

videos = sorted(os.listdir(VIDEO_DIR))
X_all  = []
y_all  = []
spk_all = []

print("Extracting DWT features...")
for i, video in enumerate(videos):
    vpath = os.path.join(VIDEO_DIR, video)
    rpath = os.path.join(ROI_DIR, video.replace('.mp4', '.txt'))

    vec = load_dwt_vector(vpath, rpath)
    if vec is not None:
        X_all.append(vec)
        y_all.append(i % 30)
        spk_all.append(i // 30 + 1)

    print(f"[{i+1}/900] {video}  dim={vec.shape if vec is not None else 'FAILED'}",
          end="\r")

X_all   = np.array(X_all)
y_all   = np.array(y_all)
spk_all = np.array(spk_all)

print(f"\nDataset: {X_all.shape}")
print(f"LL coefficients per frame: {frame_to_ll(np.zeros((250,400), np.uint8)).shape}")

# Save
df = pd.DataFrame(X_all, columns=[f"f{i}" for i in range(X_all.shape[1])])
df["y"]   = y_all
df["spk"] = spk_all
df.to_csv("dwt_lda_features.csv", index=False)


# ── 5. Train/test split ─────────────────────────────────────────

mask_train = ~np.isin(spk_all, test_speakers)
X_train = X_all[mask_train]
y_train = y_all[mask_train]
X_test  = X_all[~mask_train]
y_test  = y_all[~mask_train]

print(f"Train: {X_train.shape}  Test: {X_test.shape}")


# ── 6. Evaluate — following paper's comparison table ───────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configs = {
    # Paper Table 2 equivalents
    "DWT + PCA → SVM": Pipeline([
        ("sc",  StandardScaler()),
        ("pca", __import__('sklearn.decomposition', fromlist=['PCA']).PCA(n_components=0.95, whiten=True)),
        ("svm", SVC(kernel='rbf', C=10))
    ]),
    "DWT + LDA → SVM": Pipeline([
        ("sc",  StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(n_components=29)),
        ("svm", SVC(kernel='rbf', C=10))
    ]),
    "DWT + LDA → NB": Pipeline([
        ("sc",  StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(n_components=29)),
        ("nb",  GaussianNB())
    ]),
    "DWT + LDA → lin SVM": Pipeline([
        ("sc",  StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(n_components=29)),
        ("svm", SVC(kernel='linear', C=1))
    ]),
}

print("\nResults:")
print(f"{'Config':25s}  {'CV':>8}  {'Test':>8}  {'Train':>8}")
print("-" * 55)

for name, pipe in configs.items():
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=cv, scoring='accuracy', n_jobs=-1)
    pipe.fit(X_train, y_train)
    print(f"{name:25s}  "
          f"{scores.mean()*100:7.1f}%  "
          f"{pipe.score(X_test,  y_test)*100:7.1f}%  "
          f"{pipe.score(X_train, y_train)*100:7.1f}%")
