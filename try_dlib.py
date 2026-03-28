import cv2 as cv
import dlib
import numpy as np
import os
import pandas as pd

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def pad_frame(frame, pad=120):
    return cv.copyMakeBorder(frame, pad, pad, pad, pad, cv.BORDER_REPLICATE)

def get_lip_landmarks(frame):
    padded = pad_frame(frame)
    gray = cv.cvtColor(padded, cv.COLOR_BGR2GRAY)

    faces = detector(gray, 1)

    if len(faces) == 0:
        print("No face detected")
        return None

    shape = predictor(gray, faces[0])

    coords = np.array([[p.x, p.y] for p in shape.parts()])
    lips = coords[48:68]

    lips -= 120

    return lips

def lip_features(lips):
    # Mouth corners
    left = lips[0]
    right = lips[6]

    # Top & bottom (inner lip)
    top = lips[13]
    bottom = lips[19]

    # Width & height
    width = np.linalg.norm(right - left)
    height = np.linalg.norm(bottom - top)

    # Area & perimeter
    area = cv2.contourArea(lips)
    perimeter = cv2.arcLength(lips, True)

    # Aspect ratio
    aspect_ratio = height / (width + 1e-6)

    # Normalize (IMPORTANT)
    width_n = width
    height_n = height / width
    area_n = area / (width ** 2)
    perimeter_n = perimeter / width

    return {
        "width": width_n,
        "height": height_n,
        "area": area_n,
        "perimeter": perimeter_n,
        "aspect_ratio": aspect_ratio
    }

def draw_lips(frame, lips):
    cv.polylines(frame, [lips], True, (0, 255, 0), 2)

    for (x, y) in lips:
        cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

    return frame

def process_video(video_path, roi_path, label, spk, visualize=True):
    cap = cv.VideoCapture(video_path)
    roi = pd.read_csv(roi_path)

    features = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        row = roi.iloc[idx]
        x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])
        frame_crop = frame[y:y+h, x:x+w]
        frame_crop = cv.resize(frame_crop, None, fx=2, fy=2)
        idx += 1

        lips = get_lip_landmarks(frame_crop)
        if lips is not None:
            feats = lip_features(lips)
            feats["y"] = label
            feats["spk"] = spk
            features.append(feats)

            if visualize:
                vis = draw_lips(frame_crop.copy(), lips)
                cv.imshow("Lips", vis)

                if cv.waitKey(1) & 0xFF == 27:
                    break


    cap.release()
    cv.destroyAllWindows()
    return features

features = []

VIDEO_DIR = "./iseckane_reci_masked/videos/"
ROI_DIR = "./iseckane_reci_masked/roi/"

videos = sorted(os.listdir(VIDEO_DIR))
rows = []

i=0
for video in videos:
    video_path = os.path.join(VIDEO_DIR, video)
    roi = video.replace('.mp4', '.txt')
    roi_path = os.path.join(ROI_DIR, roi)

    ret = process_video(video_path, roi_path, i % 30, i // 30 + 1 )
    rows += ret

    i += 1
