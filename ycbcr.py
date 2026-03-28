import cv2 as cv
from cv2.typing import MatLike
import pandas as pd
import numpy as np
import random as rand
import os

def nothing(x):
    pass

WIN_NAME = "YCrCB"
HSV_WIN_NAME = "HSV"
INNER_MOUTH_WIN = "INNER MOUTH"
RGB_WIN = "RGB"

class Tester():
    def __init__(self):
        self.ly = self.lcr = self.lcb = 0
        self.hy = self.hcr = self.hcb = 0
        self.lh = self.ls = self.lv = 0
        self.hh = self.hs = self.hv = 0
        self.inner_mouth_th = 0
        self.lr = self.lg = self.lb = 0
        self.hr = self.hg = self.hb = 0

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
        
        cv.namedWindow(RGB_WIN)
        cv.createTrackbar("L - R", RGB_WIN, 0, 255, nothing)
        cv.createTrackbar("L - G", RGB_WIN, 0, 255, nothing)
        cv.createTrackbar("L - B", RGB_WIN, 0, 255, nothing)
        cv.createTrackbar("H - R", RGB_WIN, 255, 255, nothing)
        cv.createTrackbar("H - G", RGB_WIN, 255, 255, nothing)
        cv.createTrackbar("H - B", RGB_WIN, 255, 255, nothing)

    def change_to_bgr(self, image: MatLike):
        return cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    
    def draw_contour(self, frame: MatLike, contour) -> MatLike:
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

    def test_values(self, video_path: str, roi_path: str):
        roi = pd.read_csv(roi_path)
        video = cv.VideoCapture(video_path)
        idx = 0
        #print("FRAME COUNT:", int(video.get(cv.CAP_PROP_FRAME_COUNT)))
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
        
            row = roi.iloc[idx]
            roi_x, roi_y, roi_w, roi_h = \
            int(row['x']), int(row['y']), int(row['width']), int(row['height'])
        
            frame_crop = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            frame_crop = cv.medianBlur(frame_crop, 5)
            idx += 1

            #clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
        
            y_crop = cv.cvtColor(frame_crop, cv.COLOR_BGR2YCrCb)
            #y_crop[:, :, 1] = clahe.apply(y_crop[:, :, 1])
            frame_crop = cv.cvtColor(y_crop, cv.COLOR_YCrCb2BGR)
            hsv_crop = cv.cvtColor(frame_crop, cv.COLOR_BGR2HSV)
            rgb_crop = cv.cvtColor(frame_crop, cv.COLOR_BGR2RGB)
            grayscale = cv.cvtColor(frame_crop, cv.COLOR_BGR2GRAY)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    
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
        
            while True:
                self.lh = cv.getTrackbarPos("L - H", HSV_WIN_NAME)
                self.ls = cv.getTrackbarPos("L - S", HSV_WIN_NAME)
                self.lv = cv.getTrackbarPos("L - V", HSV_WIN_NAME)
                self.hh = cv.getTrackbarPos("H - H", HSV_WIN_NAME)
                self.hs = cv.getTrackbarPos("H - S", HSV_WIN_NAME)
                self.hv = cv.getTrackbarPos("H - V", HSV_WIN_NAME)
                self.ly = cv.getTrackbarPos("L - Y", WIN_NAME)
                self.lcr = cv.getTrackbarPos("L - Cr", WIN_NAME)
                self.lcb = cv.getTrackbarPos("L - Cb", WIN_NAME)
                self.hy = cv.getTrackbarPos("H - Y", WIN_NAME)
                self.hcr = cv.getTrackbarPos("H - Cr", WIN_NAME)
                self.hcb = cv.getTrackbarPos("H - Cb", WIN_NAME)
                self.inner_mouth_th = cv.getTrackbarPos("th", INNER_MOUTH_WIN)
                self.lr = cv.getTrackbarPos("L - R", RGB_WIN)
                self.lg = cv.getTrackbarPos("L - G", RGB_WIN)
                self.lb = cv.getTrackbarPos("L - B", RGB_WIN)
                self.hr = cv.getTrackbarPos("H - R", RGB_WIN)
                self.hg = cv.getTrackbarPos("H - G", RGB_WIN)
                self.hb = cv.getTrackbarPos("H - B", RGB_WIN)
        
                y_l_range = np.array([self.ly, self.lcr, self.lcb])
                y_h_range = np.array([self.hy, self.hcr, self.hcb])
                y_mask = cv.inRange(y_crop, y_l_range, y_h_range)
        
                y_mask = cv.morphologyEx(y_mask, cv.MORPH_OPEN, kernel)
                y_mask = cv.morphologyEx(y_mask, cv.MORPH_CLOSE, kernel)
    
                y_mask = cv.bitwise_and(rectangle_mask, y_mask)
        
                hsv_l_range = np.array([self.lh, self.ls, self.lv])
                hsv_h_range = np.array([self.hh, self.hs, self.hv])
                hsv_mask = cv.inRange(hsv_crop, hsv_l_range, hsv_h_range)
        
                hsv_mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, kernel)
                hsv_mask = cv.morphologyEx(hsv_mask, cv.MORPH_CLOSE, kernel)
    
                hsv_mask = cv.bitwise_and(rectangle_mask, hsv_mask)

                rgb_l_range = np.array([self.lr, self.lg, self.lb])
                rgb_h_range = np.array([self.hr, self.hg, self.hb])
                rgb_mask = cv.inRange(rgb_crop, rgb_l_range, rgb_h_range)

                rgb_mask = cv.morphologyEx(rgb_mask, cv.MORPH_OPEN, kernel)
                rgb_mask = cv.morphologyEx(rgb_mask, cv.MORPH_CLOSE, kernel)

                rgb_mask = cv.bitwise_and(rectangle_mask, rgb_mask)
        
                _, inner_mouth_mask = cv.threshold(grayscale, self.inner_mouth_th, 255, cv.THRESH_BINARY_INV)
                inner_mouth_mask = cv.morphologyEx(inner_mouth_mask, cv.MORPH_OPEN, kernel)
                inner_mouth_mask = cv.morphologyEx(inner_mouth_mask, cv.MORPH_CLOSE, kernel)

        
                #orovanje hsv i ycrcb maske i rgb
                final_mask = cv.bitwise_or(hsv_mask, y_mask)
                final_mask = cv.bitwise_or(final_mask, rgb_mask)
        
                #potrebno za kasnije razdvajanje svega
                lip_region_mask = np.zeros_like(final_mask)
    
                c, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                
                #provera da li ima kontura
                if not c:
                    cv.imshow('maska', frame_crop)
                    key = cv.waitKey(30)
                    if key == ord(' '):
                        break
                    if key == ord('q'):
                        video.release()
                        cv.destroyAllWindows()
                        return
                    continue

                
                largest = max(c, key=cv.contourArea)
                valid = [contour for contour in c 
                                  if cv.contourArea(contour) >= 0.2*cv.contourArea(largest)]
                if len(valid) >= 2:
                    s = sorted(valid, key=cv.contourArea, reverse=True)
                    cntr = np.vstack([s[0], s[1]])
                else:
                    cntr = valid[0]
                hull = cv.convexHull(cntr)
                lip_region_mask = cv.drawContours(lip_region_mask, [hull], -1, 255, cv.FILLED)
    
                #filtar koji odrzava samo unutrasnjost od usta
                inner_mouth_mask = cv.bitwise_and(lip_region_mask, inner_mouth_mask)
        
                final_mask = cv.bitwise_or(final_mask, inner_mouth_mask)
        
                final_mask = cv.morphologyEx(final_mask, cv.MORPH_OPEN, kernel)
                final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel)
        
                contours, _ = cv.findContours(
                        final_mask,
                        cv.RETR_EXTERNAL,
                        cv.CHAIN_APPROX_SIMPLE
                        )
        
                vis = self.draw_contour(frame_crop, max(contours, key=cv.contourArea))
                rgb_brg_mask = cv.cvtColor(rgb_crop, cv.COLOR_RGB2BGR)
                h_combined = np.hstack([
                    self.change_to_bgr(grayscale),
                    y_crop,
                    hsv_crop,
                    rgb_brg_mask,
                    vis
                    ])
                h_combined_2 = np.hstack([
                    self.change_to_bgr(inner_mouth_mask),
                    self.change_to_bgr(y_mask),
                    self.change_to_bgr(hsv_mask),
                    self.change_to_bgr(rgb_mask),
                    self.change_to_bgr(final_mask)
                    ])
                combined = np.vstack([h_combined, h_combined_2])
                cv.imshow('maska', combined)
                key = cv.waitKey(30)
                if key == ord(' '):
                    break
                if key == ord('q'):
                    video.release()
                    cv.destroyAllWindows()
                    return
        
        print("lower for YCrCb:",self.ly, self.lcr, self.lcb)
        print("higher for YCrCb:",self.hy, self.hcr, self.hcb)
        print("lower for hsv:",self.lh, self.ls, self.lv)
        print("higher for hsv:", self.hh, self.hs, self.hv)
        print("inner mouth threshold:", self.inner_mouth_th)
        video.release()

ranges = [
        list(range(1, 11)), 
        list(range(1, 7)), 
        list(range(1, 8)), 
        list(range(1, 8))
        ]
roi_path = "iseckane_reci_masked/roi/"
video_path = "iseckane_reci_masked/videos/"
tester = Tester()
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

        tester.test_values(video, roi)

    rows.append({
            "spk": spk,
            "ly":  tester.ly,
            "lcr": tester.lcr,
            "lcb": tester.lcb,
            "hy":  tester.hy,
            "hcr": tester.hcr,
            "hcb": tester.hcb,
            "lh":  tester.lh,
            "ls":  tester.ls,
            "lv":  tester.lv,
            "hh":  tester.hh,
            "hs":  tester.hs,
            "hv":  tester.hv,
            "lr":  tester.lr,
            "lg":  tester.lg,
            "lb":  tester.lb,
            "hr":  tester.hr,
            "hg":  tester.hg,
            "hb":  tester.hb,
            "inm_th": tester.inner_mouth_th,
            })

df = pd.DataFrame(rows)
df.to_csv("speaker_values.csv", index=False)
cv.destroyAllWindows()
