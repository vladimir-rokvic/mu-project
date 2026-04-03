#ovo je feature extractor zato se zove fe
import os
import math
from cv2.typing import MatLike
import numpy as np
import pandas as pd
import cv2 as cv
from scipy.fft import dctn
from speaker_values import speaker_df

ly = lcr = lcb = 0
hy = hcr = hcb = 0

lh = ls = lv = 0
hh = hs = hv = 0

lr = lg = lb = 0
hr = hg = hb = 0

#inner mouth mask, teeth mask
inm_mask = tht_mask = 0

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
clahe_ycrcb = cv.createCLAHE(clipLimit=1.0, tileGridSize=(3,3))
clahe_hsv = cv.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))

VIDEO_DIR = "./iseckane_reci_masked/videos/"
ROI_DIR = "./iseckane_reci_masked/roi/"

def create_hull(contour):
    return cv.convexHull(contour)

def clean_mask(mask):
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask

def set_values(row):
    global ly, lcr, lcb, hy, hcr, hcb
    global lh, ls, lv, hh, hs, hv
    global lr, lg, lb, hr, hg, hb
    global inm_mask, tht_mask

    ly = row['ly']
    lcr = row['lcr']
    lcb = row['lcb']
    hy = row['hy']
    hcr = row['hcr']
    hcb = row['hcb']

    lh = row['lh']
    ls = row['ls']
    lv = row['lv']
    hh = row['hh']
    hs = row['hs']
    hv = row['hv']

    lr = row['lr']
    lg = row['lg']
    lb = row['lb']
    hr = row['hr']
    hg = row['hg']
    hb = row['hb']

    inm_mask = row['inm_th']
    tht_mask = row['tht_th']

def get_ycrcb_mask(frame: MatLike):
    global ly, lcr, lcb, hy, hcr, hcb

    l_range = np.array([ly, lcr, lcb])
    h_range = np.array([hy, hcr, hcb])

    return cv.inRange(frame, l_range, h_range)

def get_hsv_mask(frame: MatLike):
    global lh, ls, lv, hh, hs, hv

    l_range = np.array([lh, ls, lv])
    h_range = np.array([hh, hs, hv])

    return cv.inRange(frame, l_range, h_range)

def get_rgb_mask(frame: MatLike):
    global lr, lg, lb, hr, hg, hb

    l_range = np.array([lr, lb, lg])
    h_range = np.array([hr, hb, hg])

    return cv.inRange(frame, l_range, h_range)

#kombinacija maske za unutransjost usta i zube
def get_inner_mouth_mask(frame):
    global inm_mask, tht_mask

    _, inner_mouth_mask = cv.threshold(frame, inm_mask, 255, cv.THRESH_BINARY_INV)
    _, teeth_mask = cv.threshold(frame, tht_mask, 255, cv.THRESH_BINARY)

    return cv.bitwise_or(inner_mouth_mask, teeth_mask)

def get_dct_features(frame, mask):
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    grayscale[mask == 0] = 0

    grayscale = cv.resize(grayscale, (32, 16)).astype(float)

    lip_pixels = grayscale[grayscale > 0]
    if len(lip_pixels) > 0:
        grayscale -= lip_pixels.mean()

    dct = dctn(grayscale, norm='ortho').flatten()
    dct[0] = 0

    return {f"dct_{i}": dct[i] for i in range(128)}


def process_frame(frame_crop, rectangle, roi_w, roi_h):
    y_crop = cv.cvtColor(frame_crop, cv.COLOR_BGR2YCrCb)
    hsv_crop = cv.cvtColor(frame_crop, cv.COLOR_BGR2HSV)
    rgb_crop = cv.cvtColor(frame_crop, cv.COLOR_BGR2RGB)
    grayscale = cv.cvtColor(frame_crop, cv.COLOR_BGR2GRAY)

    y_crop = cv.medianBlur(y_crop, 3)
    hsv_crop = cv.medianBlur(hsv_crop, 3)
    rgb_crop = cv.medianBlur(rgb_crop, 3)
    grayscale = cv.medianBlur(grayscale, 3)

    y_mask = get_ycrcb_mask(y_crop)
    y_mask = clean_mask(y_mask)
    y_mask = cv.bitwise_and(rectangle, y_mask)

    hsv_mask = get_hsv_mask(hsv_crop)
    hsv_mask = clean_mask(hsv_mask)
    hsv_mask = cv.bitwise_and(hsv_mask, rectangle)

    rgb_mask = get_rgb_mask(rgb_crop)
    rgb_mask = clean_mask(rgb_mask)
    rgb_mask = cv.bitwise_and(rgb_mask, rectangle)

    inner_mouth_mask = get_inner_mouth_mask(grayscale)
    inner_mouth_mask = clean_mask(inner_mouth_mask)

    lip_mask = cv.bitwise_or(y_mask, hsv_mask)
    lip_mask = cv.bitwise_or(lip_mask, rgb_mask)

    lip_region_mask = np.zeros_like(lip_mask)
    c, _ = cv.findContours(lip_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not c:
        return
    
    largest = max(c, key=cv.contourArea)
    area_threshold = 0.1*cv.contourArea(largest)
    valid = [cs for cs in c if cv.contourArea(cs) >= area_threshold]

    cntr = np.vstack(valid)
    hull = create_hull(cntr)
    lip_region_mask = cv.drawContours(lip_region_mask, [hull], -1, 255, cv.FILLED)
    lip_region_mask = clean_mask(lip_region_mask)

    inner_mouth_mask = cv.bitwise_and(lip_region_mask, inner_mouth_mask)

    final_mask = cv.bitwise_or(lip_mask, inner_mouth_mask)
    final_mask = clean_mask(final_mask)

    c, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest = max(c, key=cv.contourArea)
    area_threshold = 0.1*cv.contourArea(largest)
    valid = [cs for cs in c if cv.contourArea(cs) >= area_threshold]

    cntr = np.vstack(valid)
    hull = create_hull(cntr)

    points = hull.reshape(-1, 2)
    xs = points[:, 0]
    ys = points[:, 1]

    left_x = np.min(xs)
    right_x = np.max(xs)

    left_y = np.mean(ys[xs == left_x])
    right_y = np.mean(ys[xs == right_x])

    top_y = np.min(ys)
    bottom_y = np.max(ys)

    width = right_x - left_x
    height = bottom_y - top_y

    hull_area = cv.contourArea(hull)

    aspect_ratio = width/height if height > 0 else 0

    angle = np.degrees(np.arctan2(
            right_y - left_y,
            right_x - left_x
        ))

    (cx, cy), (MA, ma), ellipse_angle = cv.fitEllipse(hull)
    ellipse_area = math.pi / 4 * MA * ma

    M = cv.moments(hull)
    cy = M['m01'] / M['m00']
    
    upper_height = (cy - top_y) / height
    lower_height = (bottom_y - cy) / height

    obim = cv.arcLength(hull, True)
    okruglost = (4 * math.pi * hull_area) / (obim ** 2)

    feature_dict = {
            "width": width / roi_w,
            "height": height / roi_h,
            "hull_area": hull_area / (width ** 2),
            "aspect_ratio": aspect_ratio,
            "angle": angle,
            "ellipse_area": ellipse_area / (width ** 2),
            "solidity" : ellipse_area / hull_area,
            "upper_lip_height": upper_height,
            "lower_lip_height": lower_height,
            "circularity": okruglost,
            
            #ovo je samo ako mi kasnije bude bilo potrebno da izvlacim 
            #nesto iz cistog videa
            "raw_height": height,
            "raw_width": width,
            "x": left_x,
            "y": top_y
            }
    
    #y = top_y
    #x = left_x
    #for_dct = frame_crop[y:y+int(height), x:x+int(width)]
    #mask_for_dct = final_mask[y:y+int(height), x:x+int(width)]
    #dct_features = get_dct_features(for_dct, mask_for_dct)
    return feature_dict

def process_video(video_path, roi_path):
    video = cv.VideoCapture(video_path)
    roi = pd.read_csv(roi_path)

    idx = 0
    feature_frame_list = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        row = roi.iloc[idx]
        roi_x, roi_y, roi_w, roi_h = \
        int(row['x']), int(row['y']), int(row['width']), int(row['height'])
        
        frame_crop = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        frame_crop = cv.medianBlur(frame_crop, 3)
            
        idx += 1

        grayscale = cv.cvtColor(frame_crop, cv.COLOR_BGR2GRAY)
        start_x = int(0.15*roi_w)
        start_y = int(0.15*roi_h)
        end_x = roi_w - int(0.15*roi_w)
        end_y = roi_h - int(0.15*roi_h)
        rectangle_mask = cv.rectangle(
                np.zeros_like(grayscale),
                (start_x, start_y),
                (end_x, end_y),
                (255, 255, 255),
                cv.FILLED)
        
        features = process_frame(frame_crop, rectangle_mask, roi_w, roi_h)
        feature_frame_list.append(features)

    video.release()
    df = pd.DataFrame(feature_frame_list)
    return df

speaker_idx = 0

videos = sorted(os.listdir(VIDEO_DIR))
for video in videos:
    #print(video)
    row = speaker_df.iloc[speaker_idx // 30]
    set_values(row)

    roi = video.replace(".mp4", ".txt")

    video_path = os.path.join(VIDEO_DIR, video)
    roi_path = os.path.join(ROI_DIR, roi)

    df = process_video(video_path, roi_path)

    csv = video.replace(".mp4", ".csv")
    df.to_csv(os.path.join("./per_frame_features/", csv))

    speaker_idx += 1
