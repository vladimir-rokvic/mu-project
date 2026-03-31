import os
import numpy as np
import pandas as pd
import cv2 as cv
from sklearn.decomposition import PCA

VIDEO_DIR = "./iseckane_reci_masked/videos/"
ROI_DIR = "./iseckane_reci_masked/roi/"

CROP_SIZE = (64, 32)
N_COMPONENTS = 20
N_TIMESTEPS = 20
test_speakers = [4, 8, 14, 26, 28]

def load_frames(video_path: str, roi_path: str):
    video = cv.VideoCapture(video_path)
    roi = pd.read_csv(roi_path)

    frames = []

    idx = 0
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        row = roi.iloc[idx]
        x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])

        frame_crop = frame[y:y+h, x:x+w]
        grayscale = cv.cvtColor(frame_crop, cv.COLOR_BGR2GRAY)
        grayscale = cv.resize(grayscale, CROP_SIZE).astype(float)
        grayscale /= 255.0

        frames.append(grayscale.flatten())

        idx += 1

    video.release()
    return np.array(frames)

def get_row(pca, video_path, roi_path, label, spk):
    frames = load_frames(video_path, roi_path)
    frames = pca.transform(frames)

    delta = np.diff(frames, axis=0)
    delta = np.vstack([delta[0], delta])

    delta2 = np.diff(delta, axis=0)
    delta2 = np.vstack([delta2[0], delta2])

    features = np.concatenate([frames, delta, delta2], axis=1)

    ret = {}

    for i in range(features.shape[1]):
        f_old = features[:, i]

        x_old = np.linspace(0, 1, len(features))
        x_new = np.linspace(0, 1, N_TIMESTEPS)
        f_new = np.interp(x_new, x_old, f_old)

        idx = 0
        for f_val in f_new:
            name = "pca" + str(i) + "_t" + str(idx)
            ret[name] = f_val
            idx += 1

    ret["y"] = label
    ret["spk"] = spk

    return ret

videos = sorted(os.listdir(VIDEO_DIR))

train_frames = []
i = 0
for video in videos:
    video_path = os.path.join(VIDEO_DIR, video)
    roi = video.replace('.mp4', '.txt')
    roi_path = os.path.join(ROI_DIR, roi)
    
    if (i // 30 + 1) not in test_speakers:
        train_frames.append(load_frames(video_path, roi_path))

    i += 1

train_frames = np.vstack(train_frames)
pca = PCA(n_components=N_COMPONENTS, whiten=True)
pca.fit(train_frames)

rows = []
i = 0
for video in videos:
    video_path = os.path.join(VIDEO_DIR, video)
    roi = video.replace('.mp4', '.txt')
    roi_path = os.path.join(ROI_DIR, roi)
    
    rows.append(get_row(pca, video_path, roi_path, i % 30, i // 30 + 1))

    i += 1

df = pd.DataFrame(rows)
#pca_extracion ovo samo je whiten false
#pca_extraction_2 ovo samo je whiten True 64x32
#pca_extraction_3 ovo samo je whiten True i crop je 128 sa 96
#pca 4 je sa N components na 30
df.to_csv('pca_features_4.csv', index=False)
