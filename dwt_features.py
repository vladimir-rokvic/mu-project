import pywt
import cv2 as cv
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

VIDEO_DIR = "./iseckane_reci_masked/videos/"
ROI_DIR = "./iseckane_reci_masked/roi/"
CROP_SIZE = (128, 72)
test_speakers = [4, 8, 14, 26, 28]

N_TIMESTEPS = 20
N_COMPONENTS = 20

def frame_to_dwt(frame_gray, wavelet='db4', level=2):
    gray = cv.resize(frame_gray, CROP_SIZE).astype(float) / 255.0

    coeffs = pywt.wavedec2(gray, wavelet, level=level)

    # coeffs[0] = approximation (cA) — most important
    # coeffs[1] = (cH, cV, cD) detail level 2
    # coeffs[2] = (cH, cV, cD) detail level 1

    # Use approximation only — compact and noise-robust
    cA = coeffs[0].flatten()

    return cA


def load_dwt_frames(video_path, roi_path):
    video = cv.VideoCapture(video_path)
    roi = pd.read_csv(roi_path)
    frames = []
    idx = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        row = roi.iloc[idx]
        x, y, w, h = int(row['x']), int(row['y']), \
                     int(row['width']), int(row['height'])
        crop = frame[y:y+h, x:x+w]

        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        dwt  = frame_to_dwt(gray)
        frames.append(dwt)
        idx += 1

    video.release()
    return np.array(frames)

def get_row(pca, video_path, roi_path, label, spk):
    frames = load_dwt_frames(video_path, roi_path)
    frames = pca.transform(frames)

    ret = {}
    x_old = np.linspace(0, 1, len(frames))
    x_new = np.linspace(0, 1, N_TIMESTEPS)

    for i in range(N_COMPONENTS):
        f_old = frames[:, i]
        f_new = np.interp(x_new, x_old, f_old)

        idx = 0
        for f_val in f_new:
            name = "dwt" + str(i) + "_t" + str(idx)
            ret[name] = f_val
            idx += 1

    ret["y"] = label
    ret["spk"] = spk

    return ret

videos = sorted(os.listdir(VIDEO_DIR))

frames = []
i = 0
for video in videos:
    video_path = os.path.join(VIDEO_DIR, video)
    roi = video.replace('.mp4', '.txt')
    roi_path = os.path.join(ROI_DIR, roi)
    
    if (i // 30 + 1) not in test_speakers:
        frames.append(load_dwt_frames(video_path, roi_path))

    i += 1

frames = np.vstack(frames)

pca = PCA(n_components=N_COMPONENTS, whiten=True)
pca.fit(frames)

rows = []
i = 0
for video in videos:
    video_path = os.path.join(VIDEO_DIR, video)
    roi = video.replace('.mp4', '.txt')
    roi_path = os.path.join(ROI_DIR, roi)
    
    rows.append(get_row(pca, video_path, roi_path, i % 30, i // 30 + 1))

    i += 1

df = pd.DataFrame(rows)
#dwt 1 64x32
#dwt 2 128x72
df.to_csv('dwt_features_2.csv', index=False)
