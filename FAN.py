import face_alignment
import numpy as np
import cv2 as cv
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

VIDEO_DIR  = "./iseckane_reci_masked/videos/"
ROI_DIR    = "./iseckane_reci_masked/roi/"
N_TIMESTEPS  = 20
test_speakers = [4, 8, 14, 26, 28]

# Lip indices in 68-point model
# 48-59 = outer lip
# 60-67 = inner lip
OUTER_LIP = list(range(48, 60))
INNER_LIP = list(range(60, 68))
ALL_LIP   = list(range(48, 68))   # 20 points total

# ── Initialize FAN ─────────────────────────────────────────────
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device='cpu',
    face_detector='blazeface',
    #face_detector_kwargs={"filter_threshold": 0.3}  # low = more permissive
)


# ── Detection strategies ────────────────────────────────────────

def try_detect(img_rgb):
    """Try FAN detection, return landmarks or None."""
    try:
        preds = fa.get_landmarks(img_rgb)
        if preds is not None and len(preds) > 0:
            return preds[0]   # (68, 2)
    except Exception:
        pass
    return None


def get_fan_landmarks(crop_bgr):
    """
    Try multiple strategies to detect lip landmarks on mouth crop.
    Returns (20, 2) lip landmarks in crop coordinates, or None.
    """
    H, W = crop_bgr.shape[:2]
    rgb  = cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB)

    # ── Strategy 1: raw crop ────────────────────────────────────
    lms = try_detect(rgb)
    if lms is not None:
        return lms[ALL_LIP]

    # ── Strategy 2: upscale 2x ──────────────────────────────────
    up2  = cv.resize(rgb, (W*2, H*2))
    lms  = try_detect(up2)
    if lms is not None:
        pts = lms[ALL_LIP].copy()
        pts[:, 0] /= 2
        pts[:, 1] /= 2
        return pts

    # ── Strategy 3: pad into fake face ──────────────────────────
    # Mouth sits at ~65-75% height in a face
    canvas_h = H * 5
    canvas_w = W * 3
    canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    y_off    = int(canvas_h * 0.65)
    x_off    = int((canvas_w - W) / 2)

    canvas[y_off:y_off+H, x_off:x_off+W] = crop_bgr
    canvas_rgb = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)

    lms = try_detect(canvas_rgb)
    if lms is not None:
        pts = lms[ALL_LIP].copy()
        pts[:, 0] -= x_off
        pts[:, 1] -= y_off
        return pts

    # ── Strategy 4: pad + upscale ───────────────────────────────
    canvas_up  = cv.resize(canvas_rgb, (canvas_w*2, canvas_h*2))
    lms        = try_detect(canvas_up)
    if lms is not None:
        pts = lms[ALL_LIP].copy()
        pts[:, 0] = pts[:, 0] / 2 - x_off
        pts[:, 1] = pts[:, 1] / 2 - y_off
        return pts

    return None


# ── Extract geometric features from landmarks ───────────────────

def landmark_features(lms, frame_h, frame_w):
    """
    lms: (20, 2) — lip landmark coordinates in crop space
    Returns normalized geometric feature dict.
    """
    xs = lms[:, 0]
    ys = lms[:, 1]

    # Key points
    left_x  = xs.min();  left_y  = ys[xs.argmin()]
    right_x = xs.max();  right_y = ys[xs.argmax()]
    top_y   = ys.min();  top_x   = xs[ys.argmin()]
    bot_y   = ys.max();  bot_x   = xs[ys.argmax()]

    width  = right_x - left_x + 1e-6
    height = bot_y   - top_y  + 1e-6

    # Centroid
    cx = xs.mean()
    cy = ys.mean()

    # Outer lip points (48-59 → indices 0-11 in ALL_LIP)
    outer = lms[:12]
    # Inner lip points (60-67 → indices 12-19 in ALL_LIP)
    inner = lms[12:]

    # Inner mouth opening — vertical distance between inner lips
    if len(inner) > 0:
        inner_ys = inner[:, 1]
        inner_top = inner[np.argmin(inner_ys)]
        inner_bot = inner[np.argmax(inner_ys)]
        opening   = (inner_ys.max() - inner_ys.min()) / (height + 1e-6)
    else:
        inner_top = np.array([cx, cy])
        inner_bot = np.array([cx, cy])
        opening   = 0.0

    # Lip curvature — distance from top point to corner line
    #dx  = right_x - left_x
    #dy  = right_y - left_y
    #curvature = abs(dy*(top_x - left_x) - dx*(top_y - left_y)) / \
    #            (np.sqrt(dx**2 + dy**2) * width + 1e-6)

    return {
        # Shape
        "lm_height":        height / width,
        "lm_aspect_ratio":  width  / (height + 1e-6),
        "lm_opening":       opening,

        # Corners
        "lm_corner_angle":  np.degrees(np.arctan2(right_y - left_y,
                                                    right_x - left_x)),
        #"lm_corner_diff":   abs(left_y - right_y) / (width + 1e-6),

        # Position
        #"lm_centroid_x":    cx / frame_w,
        #"lm_centroid_y":    cy / frame_h,
        "lm_upper_ratio":   (cy - top_y) / (height + 1e-6),
        "lm_lower_ratio":   (bot_y - cy) / (height + 1e-6),

        # Shape descriptors
        #"lm_curvature":     curvature,
        "lm_width_norm":    width  / frame_w,
        "lm_height_norm":   height / frame_h,

        # Normalized landmark coords — full shape descriptor
        # Subtract centroid and normalize by width
        #**{f"lm_x{i}": (lms[i, 0] - cx) / width  for i in range(20)},
        #**{f"lm_y{i}": (lms[i, 1] - cy) / (height + 1e-6) for i in range(20)},
    }


# ── Per-video extraction ────────────────────────────────────────

def extract_landmark_features(video_path, roi_path, label, spk):
    video  = cv.VideoCapture(video_path)
    roi    = pd.read_csv(roi_path)
    idx    = 0
    rows   = []
    n_fail = 0

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

        lms = get_fan_landmarks(crop)

        if lms is not None:
            feats = landmark_features(lms, h, w)
            rows.append(feats)
        else:
            n_fail += 1

        idx += 1

    video.release()

    if len(rows) == 0:
        return None, n_fail

    return pd.DataFrame(rows), n_fail   # (T, n_features)


def aggregate_video(frame_df, label, spk):
    """
    Aggregate per-frame landmark features to fixed-size video vector.
    Uses mean/std/max/min + delta.
    """
    arr   = frame_df.values.astype(float)
    #T     = len(arr)
    #x_old = np.linspace(0, 1, T)
    #x_new = np.linspace(0, 1, N_TIMESTEPS)

    ret = {}
    for fi, col in enumerate(frame_df.columns):
        series = arr[:, fi]
        #interp = np.interp(x_new, x_old, series)

        #for t, val in enumerate(interp):
        #    ret[f"{col}_t{t}"] = val

        ret[f"{col}_mean"]  = series.mean()
        ret[f"{col}_std"]   = series.std()
        ret[f"{col}_max"]   = series.max()
        ret[f"{col}_min"]   = series.min()
        ret[f"{col}_range"] = series.max() - series.min()

    ret["y"]   = label
    ret["spk"] = spk
    return ret


# ── Main ────────────────────────────────────────────────────────

videos    = sorted(os.listdir(VIDEO_DIR))
rows      = []
total_fail = 0
total_frames = 0

print("Extracting FAN landmark features...")
print("Controls: each video logs detection rate\n")

for i, video_name in enumerate(videos):
    vpath = os.path.join(VIDEO_DIR, video_name)
    rpath = os.path.join(ROI_DIR, video_name.replace('.mp4', '.txt'))

    frame_df, n_fail = extract_landmark_features(
        vpath, rpath, i % 30, i // 30 + 1
    )

    if frame_df is not None and len(frame_df) > 0:
        row = aggregate_video(frame_df, i % 30, i // 30 + 1)
        rows.append(row)
        n_frames    = len(frame_df) + n_fail
        total_fail += n_fail
        total_frames += n_frames
        det_rate    = (n_frames - n_fail) / (n_frames + 1e-6) * 100
        print(f"[{i+1:3d}/900] {video_name}  "
              f"frames={n_frames}  "
              f"detected={n_frames-n_fail}  "
              f"rate={det_rate:.0f}%")
    else:
        print(f"[{i+1:3d}/900] {video_name}  COMPLETELY FAILED")

# ── Summary ─────────────────────────────────────────────────────
overall_rate = (total_frames - total_fail) / (total_frames + 1e-6) * 100
print(f"\nOverall detection rate: {overall_rate:.1f}%")
print(f"Total frames: {total_frames}  Failed: {total_fail}")

df = pd.DataFrame(rows)
df.to_csv("fan_landmark_features.csv", index=False)
print(f"Saved fan_landmark_features.csv  shape={df.shape}")

# Sanity check
if "y" in df.columns and "spk" in df.columns:
    print(df["y"].value_counts().sort_index())
    print(df["spk"].value_counts().sort_index())
