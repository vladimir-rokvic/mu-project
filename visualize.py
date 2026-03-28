import os
import numpy as np
import pandas as pd
from new_e import create_hull
import cv2 as cv

speaker_df = pd.read_csv('./speaker_values2.csv')

ly = lcr = lcb = 0
hy = hcr = hcb = 0
lh = ls = lv = 0
hh = hs = hv = 0
lr = lg = lb = 0
hr = hg = hb = 0
inm_mask = tht_mask = 0

kernel  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

VIDEO_DIR = "./iseckane_reci_masked/videos/"
ROI_DIR   = "./iseckane_reci_masked/roi/"

def clean_mask(mask):
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask

def set_values(row):
    global ly, lcr, lcb, hy, hcr, hcb
    global lh, ls, lv, hh, hs, hv
    global lr, lg, lb, hr, hg, hb
    global inm_mask, tht_mask
    ly=row['ly']; lcr=row['lcr']; lcb=row['lcb']
    hy=row['hy']; hcr=row['hcr']; hcb=row['hcb']
    lh=row['lh']; ls=row['ls'];   lv=row['lv']
    hh=row['hh']; hs=row['hs'];   hv=row['hv']
    lr=row['lr']; lg=row['lg'];   lb=row['lb']
    hr=row['hr']; hg=row['hg'];   hb=row['hb']
    inm_mask=row['inm_th']; tht_mask=row['tht_th']

def get_ycrcb_mask(frame):
    return cv.inRange(frame,
                      np.array([ly,  lcr, lcb]),
                      np.array([hy,  hcr, hcb]))

def get_hsv_mask(frame):
    return cv.inRange(frame,
                      np.array([lh,  ls,  lv]),
                      np.array([hh,  hs,  hv]))

def get_rgb_mask(frame):
    return cv.inRange(frame,
                      np.array([lr,  lb,  lg]),
                      np.array([hr,  hb,  hg]))

def get_inner_mouth_mask(frame):
    _, inner = cv.threshold(frame, inm_mask, 255, cv.THRESH_BINARY_INV)
    _, teeth = cv.threshold(frame, tht_mask, 255, cv.THRESH_BINARY)
    return cv.bitwise_or(inner, teeth)


def process_frame(frame_crop, rectangle):
    y_crop   = cv.medianBlur(cv.cvtColor(frame_crop, cv.COLOR_BGR2YCrCb), 3)
    hsv_crop = cv.medianBlur(cv.cvtColor(frame_crop, cv.COLOR_BGR2HSV),   3)
    rgb_crop = cv.medianBlur(cv.cvtColor(frame_crop, cv.COLOR_BGR2RGB),   3)
    gray     = cv.medianBlur(cv.cvtColor(frame_crop, cv.COLOR_BGR2GRAY),  3)

    y_mask   = cv.bitwise_and(rectangle, clean_mask(get_ycrcb_mask(y_crop)))
    hsv_mask = cv.bitwise_and(rectangle, clean_mask(get_hsv_mask(hsv_crop)))
    rgb_mask = cv.bitwise_and(rectangle, clean_mask(get_rgb_mask(rgb_crop)))
    inner    = clean_mask(get_inner_mouth_mask(gray))

    lip_mask = cv.bitwise_or(cv.bitwise_or(y_mask, hsv_mask), rgb_mask)

    c, _ = cv.findContours(lip_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not c:
        return None

    largest   = max(c, key=cv.contourArea)
    threshold = 0.1 * cv.contourArea(largest)
    valid     = [cs for cs in c if cv.contourArea(cs) >= threshold]
    hull      = create_hull(np.vstack(valid))

    lip_region = np.zeros_like(lip_mask)
    cv.drawContours(lip_region, [hull], -1, 255, cv.FILLED)
    lip_region = clean_mask(lip_region)

    # Inner mouth = dark region inside lip hull
    inner_inside = cv.bitwise_and(lip_region, inner)

    final_mask = clean_mask(cv.bitwise_or(lip_mask, inner_inside))

    c, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not c:
        return None

    largest   = max(c, key=cv.contourArea)
    threshold = 0.1 * cv.contourArea(largest)
    valid     = [cs for cs in c if cv.contourArea(cs) >= threshold]
    hull_final = create_hull(np.vstack(valid))

    # Compute opening
    hull_fill = np.zeros_like(inner)
    cv.drawContours(hull_fill, [hull_final], -1, 255, cv.FILLED)
    inner_region = cv.bitwise_and(hull_fill, inner)
    opening_px   = cv.countNonZero(inner_region)

    leftmost  = tuple(hull_final[hull_final[:,:,0].argmin()][0])
    rightmost = tuple(hull_final[hull_final[:,:,0].argmax()][0])
    width     = rightmost[0] - leftmost[0]
    opening   = opening_px / (width ** 2 + 1e-6)

    return {
        "hull":          hull_final,
        "lip_mask":      lip_mask,
        "inner_raw":     inner,
        "inner_region":  inner_region,   # inner mouth INSIDE hull only
        "final_mask":    final_mask,
        "opening":       opening,
        "opening_px":    opening_px,
        "width":         width,
    }


def visualize_video(video_path, roi_path):
    video = cv.VideoCapture(video_path)
    roi   = pd.read_csv(roi_path)
    idx   = 0

    print("SPACE = next frame | Q = next video | ESC = quit")

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        row = roi.iloc[idx]
        rx, ry, rw, rh = int(row['x']), int(row['y']), \
                         int(row['width']), int(row['height'])
        crop = cv.medianBlur(frame[ry:ry+rh, rx:rx+rw], 5)
        idx += 1

        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        sx = int(0.14*rw);  ex = rw - sx
        sy = int(0.14*rh);  ey = rh - sy
        rect_mask = cv.rectangle(
            np.zeros_like(gray),
            (sx, sy), (ex, ey), 255, cv.FILLED)

        result = process_frame(crop, rect_mask)

        # ── Build panels ────────────────────────────────────────
        def to_bgr(m):
            return cv.cvtColor(m, cv.COLOR_GRAY2BGR)

        if result is not None:
            hull         = result["hull"]
            inner_raw    = result["inner_raw"]
            inner_region = result["inner_region"]
            final_mask   = result["final_mask"]
            opening      = result["opening"]
            opening_px   = result["opening_px"]

            # Panel 1 — original crop
            p1 = crop.copy()

            # Panel 2 — hull outline on crop
            p2 = crop.copy()
            cv.drawContours(p2, [hull], -1, (0, 255, 0), 2)

            # Panel 3 — inner mouth mask (raw, before hull clip)
            p3 = to_bgr(inner_raw)
            cv.putText(p3, "inner raw", (3, 12),
                       cv.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)

            # Panel 4 — inner mouth region INSIDE hull (what we measure)
            p4 = to_bgr(inner_region)
            cv.putText(p4, "inner region", (3, 12),
                       cv.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)

            # Panel 5 — overlay: hull + inner region highlighted
            p5 = crop.copy()
            # Green tint over inner region
            overlay = p5.copy()
            overlay[inner_region > 0] = (0, 220, 0)
            cv.addWeighted(overlay, 0.45, p5, 0.55, 0, p5)
            cv.drawContours(p5, [hull], -1, (255, 255, 0), 2)
            cv.putText(p5, f"open={opening:.3f} px={opening_px}",
                       (3, 12), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

            # Panel 6 — final mask
            p6 = to_bgr(final_mask)
            cv.putText(p6, "final mask", (3, 12),
                       cv.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)

        else:
            blank = np.zeros_like(crop)
            cv.putText(blank, "NO DETECTION", (5, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            p1 = p2 = p3 = p4 = p5 = p6 = blank

        # Resize all panels to same height
        target_h = 160
        def rh_img(img):
            scale = target_h / img.shape[0]
            return cv.resize(img, (int(img.shape[1]*scale), target_h))

        row1 = np.hstack([rh_img(p1), rh_img(p2), rh_img(p3)])
        row2 = np.hstack([rh_img(p4), rh_img(p5), rh_img(p6)])

        # Labels
        labels_r1 = ["Original", "Hull", "Inner Raw"]
        labels_r2 = ["Inner Region", "Opening Overlay", "Final Mask"]
        w_each = row1.shape[1] // 3

        for j, lbl in enumerate(labels_r1):
            cv.putText(row1, lbl, (j*w_each + 4, target_h - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1)
        for j, lbl in enumerate(labels_r2):
            cv.putText(row2, lbl, (j*w_each + 4, target_h - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1)

        combined = np.vstack([row1, row2])
        cv.imshow("Inner Mouth Visualization", combined)

        key = cv.waitKey(0)
        if key == ord('q'):
            video.release()
            return False
        elif key == 27:   # ESC
            video.release()
            return True

    video.release()
    return False


# ── Main ────────────────────────────────────────────────────────

videos = sorted(os.listdir(VIDEO_DIR))

for i, video_name in enumerate(videos):
    row = speaker_df.iloc[i // 30]
    set_values(row)

    video_path = os.path.join(VIDEO_DIR, video_name)
    roi_path   = os.path.join(ROI_DIR, video_name.replace(".mp4", ".txt"))

    print(f"\n[{i+1}/900] {video_name}  spk={i//30+1}")
    quit_all = visualize_video(video_path, roi_path)

    if quit_all:
        break

cv.destroyAllWindows()
