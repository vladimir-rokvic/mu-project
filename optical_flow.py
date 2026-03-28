import numpy as np
import cv2 as cv
import pandas as pd
import os

VIDEO_DIR = "./iseckane_reci_masked/videos/"
ROI_DIR = "./iseckane_reci_masked/roi/"

CROP_SIZE = (128, 96)
N_TIMESTEPS = 20

def get_row(video_path: str, roi_path: str, label, spk):
    video = cv.VideoCapture(video_path)
    roi = pd.read_csv(roi_path)

    flow_features = []

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

        features = np.array([mag.mean(), 
                             mag.std(), 
                             mag.max(), 
                             flow[..., 0].mean(),
                             flow[..., 1].mean(),
                             ang.std()])

        flow_features.append(features)

        prev = grayscale
        idx += 1

    video.release()
    flow_features = np.array(flow_features)

    x_old = np.linspace(0, 1, len(flow_features))
    x_new = np.linspace(0, 1, N_TIMESTEPS)

    ret_row = {}

    for i in range(6):
        f_old = flow_features[:, i]
        f_new = np.interp(x_new, x_old, f_old)

        idx = 0
        for f_val in f_new:
            name = "flow_" + str(i) + "_t" + str(idx)
            ret_row[name] = f_val
            idx += 1

    ret_row["y"] = label
    ret_row["spk"] = spk
    return ret_row

videos = sorted(os.listdir(VIDEO_DIR))

i = 0
rows = []
for video in videos:
    video_path = os.path.join(VIDEO_DIR, video)
    roi = video.replace('.mp4', '.txt')
    roi_path = os.path.join(ROI_DIR, roi)
    
    rows.append(get_row(video_path, roi_path, i % 30, i // 30 + 1))

    i += 1

df = pd.DataFrame(rows)
#optical flow w:200 h:150 timesteps:20
#optical flow 2 w:128 h:96 timesteps:40
#optical flow 3 w:128 h:96 timesteps:20
df.to_csv('optical_flow_2.csv', index=False)
