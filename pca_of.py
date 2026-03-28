import os
import cv2 as cv
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CROP_SIZE = (64, 32)
N_TIMESTEPS = 20
N_COMPONENTS = 20

VIDEO_DIR = "./iseckane_reci_masked/videos/"
ROI_DIR = "./iseckane_reci_masked/roi/"

test_speakers = [4, 8, 14, 26, 28]

def load_flows(video_path, roi_path):
    video = cv.VideoCapture(video_path)
    roi = pd.read_csv(roi_path)

    flow_frames= []

    idx = 0

    row = roi.iloc[idx]
    x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])
    idx += 1

    ret, prev = video.read()
    if not ret:
        return

    prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    prev = prev[y:y+h, x:x+w]
    prev = cv.resize(prev, CROP_SIZE)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        row = roi.iloc[idx]
        x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])

        frame_crop = frame[y:y+h, x:x+w]

        grayscale = cv.cvtColor(frame_crop, cv.COLOR_BGR2GRAY)
        grayscale = cv.resize(grayscale, CROP_SIZE)

        flow = cv.calcOpticalFlowFarneback(prev, grayscale, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        
        mag = mag.flatten()
        ang = ang.flatten()

        features = np.concatenate([mag, ang])
        flow_frames.append(features)

        prev = grayscale
        idx += 1

    video.release()

    return np.array(flow_frames)

def load_flow_separate(video_path, roi_path):
    video = cv.VideoCapture(video_path)
    roi = pd.read_csv(roi_path)

    mag_frames = []
    ang_frames = []

    idx = 0

    row = roi.iloc[idx]
    x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])
    idx += 1

    ret, prev = video.read()
    if not ret:
        return

    prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    prev = prev[y:y+h, x:x+w]
    prev = cv.resize(prev, CROP_SIZE)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        row = roi.iloc[idx]
        x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])

        frame_crop = frame[y:y+h, x:x+w]

        grayscale = cv.cvtColor(frame_crop, cv.COLOR_BGR2GRAY)
        grayscale = cv.resize(grayscale, CROP_SIZE)

        flow = cv.calcOpticalFlowFarneback(prev, grayscale, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        
        mag = mag.flatten()
        ang = ang.flatten()

        mag_frames.append(mag)
        ang_frames.append(ang)

        prev = grayscale
        idx += 1

    video.release()

    return np.array(mag_frames), np.array(ang_frames)

def get_row(pca, sc, video_path, roi_path, label, spk):
    frames = load_flows(video_path, roi_path)
    frames = sc.transform(frames)
    frames = pca.transform(frames)

    ret = {}

    for i in range(N_COMPONENTS):
        f_old = frames[:, i]

        x_old = np.linspace(0, 1, len(frames))
        x_new = np.linspace(0, 1, N_TIMESTEPS)
        f_new = np.interp(x_new, x_old, f_old)

        idx = 0
        for f_val in f_new:
            name = "pca_" + str(i) + "_t" + str(idx)
            ret[name] = f_val
            idx += 1

    ret["y"] = label
    ret["spk"] = spk
    return ret

def get_row_separate(pca_mag, pca_ang, sc_mag, sc_ang, video_path, roi_path, label, spk):
    mag_frames, ang_frames = load_flow_separate(video_path, roi_path)
    mag_frames, ang_frames = sc_mag.transform(mag_frames), sc_ang.transform(ang_frames)
    mag_frames, ang_frames = pca_mag.transform(mag_frames), pca_ang.transform(ang_frames)

    ret_mag = {}
    ret_ang = {}

    for i in range(N_COMPONENTS):
        mag_f_old = mag_frames[:, i]
        ang_f_old = ang_frames[:, i]

        x_old = np.linspace(0, 1, len(mag_frames))
        x_new = np.linspace(0, 1, N_TIMESTEPS)

        mag_f_new = np.interp(x_new, x_old, mag_f_old)
        ang_f_new = np.interp(x_new, x_old, ang_f_old)

        idx = 0
        for f_val in mag_f_new:
            mag_name = "mag_pca_" + str(i) + "_t" + str(idx)
            ret_mag[mag_name] = f_val
            idx += 1

        idx = 0
        for f_val in ang_f_new:
            ang_name = "ang_pca_" + str(i) + "_t" + str(idx)
            ret_ang[ang_name] = f_val
            idx += 1

    ret_mag["y"] = label
    ret_mag["spk"] = spk

    ret_ang["y"] = label
    ret_ang["spk"] = spk
    return ret_mag, ret_ang

videos = sorted(os.listdir(VIDEO_DIR))

i = 0
mag_frames = []
ang_frames = []
for video in videos:
    video_path = os.path.join(VIDEO_DIR, video)
    roi = video.replace('.mp4', '.txt')
    roi_path = os.path.join(ROI_DIR, roi)
    
    if (i // 30 + 1) not in test_speakers:
        mag, ang = load_flow_separate(video_path, roi_path)
        mag_frames.append(mag)
        ang_frames.append(ang)

    i += 1

mag_frames = np.vstack(mag_frames)
ang_frames = np.vstack(ang_frames)

sc_mag = StandardScaler()
sc_ang = StandardScaler()
mag_frames = sc_mag.fit_transform(mag_frames)
ang_frames = sc_ang.fit_transform(ang_frames)

pca_mag = PCA(n_components=N_COMPONENTS, whiten=True)
pca_ang = PCA(n_components=N_COMPONENTS, whiten=True)
pca_mag.fit(mag_frames)
pca_ang.fit(ang_frames)

rows_mag = []
rows_ang = []
i = 0
for video in videos:
    video_path = os.path.join(VIDEO_DIR, video)
    roi = video.replace('.mp4', '.txt')
    roi_path = os.path.join(ROI_DIR, roi)
    
    mag, ang = get_row_separate(pca_mag, pca_ang, sc_mag, sc_ang, video_path, roi_path, i % 30, i // 30 + 1)
    rows_mag.append(mag)
    rows_ang.append(ang)

    i += 1

mag_df = pd.DataFrame(rows_mag)
ang_df = pd.DataFrame(rows_ang)
mag_df.to_csv('mag_features.csv', index=False)
ang_df.to_csv('ang_features.csv', index=False)
