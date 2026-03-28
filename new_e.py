import cv2 as cv
from cv2.typing import MatLike
import pandas as pd
import numpy as np
import random as rand
import os

def nothing(x):
    pass

def change_to_bgr(image: MatLike):
    return cv.cvtColor(image, cv.COLOR_GRAY2BGR)

def change_to_gray(image: MatLike):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def draw_contour(frame: MatLike, contour) -> MatLike:
    vis = frame.copy()
    points = contour.reshape(-1, 2)
    xs = points[:, 0]
    ys = points[:, 1]
    left_x = np.min(xs)
    right_x = np.max(xs)
    left_y = np.mean(ys[xs == left_x])
    right_y = np.mean(ys[xs == right_x])
    top_y = np.min(ys)
    bottom_y = np.max(ys)
    width = right_x - left_x
    cv.drawContours(vis, [contour], -1, (0,255,0), 2)
    cv.circle(vis, (int(left_x), int(left_y)), 5, (255,0,0), -1)
    cv.circle(vis, (int(right_x), int(right_y)), 5, (0,0,255), -1)
    cv.circle(vis, (left_x + int(width/2), int(top_y)), 5, (0,255,255), -1)
    cv.circle(vis, (left_x + int(width/2), int(bottom_y)), 5, (255,255,0), -1)
    return vis

def get_ycrcb_mask(frame: MatLike):
    global ly, lcr, lcb, hy, hcr, hcb
    ly = cv.getTrackbarPos("L - Y", WIN_NAME)
    lcr = cv.getTrackbarPos("L - Cr", WIN_NAME)
    lcb = cv.getTrackbarPos("L - Cb", WIN_NAME)
    hy = cv.getTrackbarPos("H - Y", WIN_NAME)
    hcr = cv.getTrackbarPos("H - Cr", WIN_NAME)
    hcb = cv.getTrackbarPos("H - Cb", WIN_NAME)

    l_range = np.array([ly, lcr, lcb])
    h_range = np.array([hy, hcr, hcb])

    return cv.inRange(frame, l_range, h_range)

def get_hsv_mask(frame: MatLike):
    global lh, ls, lv, hh, hs, hv
    lh = cv.getTrackbarPos("L - H", HSV_WIN_NAME)
    ls = cv.getTrackbarPos("L - S", HSV_WIN_NAME)
    lv = cv.getTrackbarPos("L - V", HSV_WIN_NAME)
    hh = cv.getTrackbarPos("H - H", HSV_WIN_NAME)
    hs = cv.getTrackbarPos("H - S", HSV_WIN_NAME)
    hv = cv.getTrackbarPos("H - V", HSV_WIN_NAME)

    l_range = np.array([lh, ls, lv])
    h_range = np.array([hh, hs, hv])

    return cv.inRange(frame, l_range, h_range)

def get_rgb_mask(frame: MatLike):
    global lr, lg, lb, hr, hg, hb
    lr = cv.getTrackbarPos("L - R", RGB_WIN)
    lg = cv.getTrackbarPos("L - G", RGB_WIN)
    lb = cv.getTrackbarPos("L - B", RGB_WIN)
    hr = cv.getTrackbarPos("H - R", RGB_WIN)
    hg = cv.getTrackbarPos("H - G", RGB_WIN)
    hb = cv.getTrackbarPos("H - B", RGB_WIN)

    l_range = np.array([lr, lb, lg])
    h_range = np.array([hr, hb, hg])

    return cv.inRange(frame, l_range, h_range)

#kombinacija maske za unutransjost usta i zube
def get_inner_mouth_mask(frame):
    global inm_mask, tht_mask
    inm_mask = cv.getTrackbarPos("th", INNER_MOUTH_WIN)
    tht_mask = cv.getTrackbarPos("tth", INNER_MOUTH_WIN)

    _, inner_mouth_mask = cv.threshold(frame, inm_mask, 255, cv.THRESH_BINARY_INV)
    _, teeth_mask = cv.threshold(frame, tht_mask, 255, cv.THRESH_BINARY)

    return cv.bitwise_or(inner_mouth_mask, teeth_mask)

def create_hull(contour):
    return cv.convexHull(contour)

def clean_mask(mask):
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask

def proccess_frame(frame_crop, rectangle):
    y_crop = cv.cvtColor(frame_crop, cv.COLOR_BGR2YCrCb)
    hsv_crop = cv.cvtColor(frame_crop, cv.COLOR_BGR2HSV)
    rgb_crop = cv.cvtColor(frame_crop, cv.COLOR_BGR2RGB)
    grayscale = cv.cvtColor(frame_crop, cv.COLOR_BGR2GRAY)

    y_crop =    cv.medianBlur(y_crop,   3)
    hsv_crop =  cv.medianBlur(hsv_crop, 3)
    rgb_crop =  cv.medianBlur(rgb_crop, 3)
    grayscale = cv.medianBlur(grayscale,3)

    while True:
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
            cv.imshow('maska_losa', frame_crop)
            key = cv.waitKey(30)
            if key == ord(' '):
                break
            if key == ord('q'):
                video.release()
                cv.destroyAllWindows()
                return
            continue
        
        largest = max(c, key=cv.contourArea)
        area_threshold = 0.1*cv.contourArea(largest)
        valid = [cs for cs in c if cv.contourArea(cs) >= area_threshold]

        #if len(valid) >= 2:
        #    s = sorted(valid, key=cv.contourArea, reverse=True)
        #    cntr = np.vstack([s[0], s[1]])
        #else:
        #    cntr = valid[0]
        cntr = np.vstack(valid)
        hull = create_hull(cntr)
        lip_region_mask = cv.drawContours(lip_region_mask, [hull], -1, 255, cv.FILLED)
        lip_region_mask = clean_mask(lip_region_mask)

        inner_mouth_mask = cv.bitwise_and(lip_region_mask, inner_mouth_mask)
        #inner_mouth_mask = cv.bitwise_xor(lip_mask, inner_mouth_mask)

        final_mask = cv.bitwise_or(lip_mask, inner_mouth_mask)
        final_mask = clean_mask(final_mask)

        c, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        largest = max(c, key=cv.contourArea)
        area_threshold = 0.1*cv.contourArea(largest)
        valid = [i for i in c if cv.contourArea(i) >= area_threshold]

        valid = np.vstack(valid)
        hull = create_hull(valid)

        vis = draw_contour(frame_crop, hull)
        rgb_brg_mask = cv.cvtColor(rgb_crop, cv.COLOR_RGB2BGR)
        h_combined = np.hstack([
            change_to_bgr(grayscale),
            y_crop,
            hsv_crop,
            rgb_brg_mask,
            vis
            ])
        h_combined_2 = np.hstack([
            change_to_bgr(inner_mouth_mask),
            change_to_bgr(y_mask),
            change_to_bgr(hsv_mask),
            change_to_bgr(rgb_mask),
            change_to_bgr(final_mask)
            ])
        combined = np.vstack([h_combined, h_combined_2])
        cv.imshow('maska', combined)
        key = cv.waitKey(30)
        if key == ord(' '):
            break
        if key == ord('q'):
            cv.destroyAllWindows()
            exit(0)

def test_video(video_path: str, roi_path: str):
        roi = pd.read_csv(roi_path)
        video = cv.VideoCapture(video_path)
        idx = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
        
            row = roi.iloc[idx]
            roi_x, roi_y, roi_w, roi_h = \
            int(row['x']), int(row['y']), int(row['width']), int(row['height'])
        
            frame_crop = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

            #frame_crop = cv.cvtColor(frame_crop, cv.COLOR_BGR2YCrCb)
            #y, cr, cb = cv.split(frame_crop)
            #cr = clahe_ycrcb.apply(cr)
            #frame_crop = cv.merge((y, cr, cb))
            #frame_crop = cv.cvtColor(frame_crop, cv.COLOR_YCrCb2BGR)

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
            
            proccess_frame(frame_crop, rectangle_mask)

        video.release()

if __name__ == "__main__":
    WIN_NAME = "YCrCB"
    HSV_WIN_NAME = "HSV"
    INNER_MOUTH_WIN = "INNER MOUTH"
    RGB_WIN = "RGB"
    
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

    cv.namedWindow(WIN_NAME)
    cv.createTrackbar("L - Y", WIN_NAME, 0, 255, nothing)
    cv.createTrackbar("L - Cr", WIN_NAME, 0, 255, nothing)
    cv.createTrackbar("L - Cb", WIN_NAME, 0, 255, nothing)
    cv.createTrackbar("H - Y", WIN_NAME, 255, 255, nothing)
    cv.createTrackbar("H - Cr", WIN_NAME, 255, 255, nothing)
    cv.createTrackbar("H - Cb", WIN_NAME, 255, 255, nothing)
    
    cv.namedWindow(HSV_WIN_NAME)
    cv.createTrackbar("L - H", HSV_WIN_NAME, 0, 179, nothing)
    cv.createTrackbar("L - S", HSV_WIN_NAME, 0, 255, nothing)
    cv.createTrackbar("L - V", HSV_WIN_NAME, 0, 255, nothing)
    cv.createTrackbar("H - H", HSV_WIN_NAME, 179, 179, nothing)
    cv.createTrackbar("H - S", HSV_WIN_NAME, 255, 255, nothing)
    cv.createTrackbar("H - V", HSV_WIN_NAME, 255, 255, nothing)
    
    cv.namedWindow(INNER_MOUTH_WIN)
    cv.createTrackbar("th", INNER_MOUTH_WIN, 0, 255, nothing)
    cv.createTrackbar("tth", INNER_MOUTH_WIN, 0, 255, nothing)
    
    cv.namedWindow(RGB_WIN)
    cv.createTrackbar("L - R", RGB_WIN, 0, 255, nothing)
    cv.createTrackbar("L - G", RGB_WIN, 0, 255, nothing)
    cv.createTrackbar("L - B", RGB_WIN, 0, 255, nothing)
    cv.createTrackbar("H - R", RGB_WIN, 255, 255, nothing)
    cv.createTrackbar("H - G", RGB_WIN, 255, 255, nothing)
    cv.createTrackbar("H - B", RGB_WIN, 255, 255, nothing)
    
    ranges = [
            list(range(1, 11)), 
            list(range(1, 7)), 
            list(range(1, 8)), 
            list(range(1, 8))
            ]
    roi_path = "iseckane_reci_masked/roi/"
    video_path = "iseckane_reci_masked/videos/"
    rows = []
    for i in range(1, 31):
        if i < 10:
            spk = "0" + str(i)
        else:
            spk = str(i)
    
        for _ in range(6):
            idx = rand.randint(1, 4)
            r = rand.sample(ranges[idx-1], 1)[0]
            if r >= 10:
                r_str = str(r)
            else:
                r_str = "0" + str(r)
    
            s = "spk" + spk + "_" + "00" + str(idx) + "_" + r_str
            roi = os.path.join(roi_path, s + ".txt")
            video = os.path.join(video_path, s + ".mp4")
            if not os.path.exists(roi) or not os.path.exists(video):
                print(f"Skipping {s} — file not found")
                continue
    
            test_video(video, roi)
    
        rows.append({
                "spk": spk,
                "ly":  ly,
                "lcr": lcr,
                "lcb": lcb,
                "hy":  hy,
                "hcr": hcr,
                "hcb": hcb,
                "lh":  lh,
                "ls":  ls,
                "lv":  lv,
                "hh":  hh,
                "hs":  hs,
                "hv":  hv,
                "lr":  lr,
                "lg":  lg,
                "lb":  lb,
                "hr":  hr,
                "hg":  hg,
                "hb":  hb,
                "inm_th": inm_mask,
                "tht_th": tht_mask
                })
    
    df = pd.DataFrame(rows)
    df.to_csv("speaker_values4.csv", index=False)
    cv.destroyAllWindows()
