import sys
import os
import traceback

# --- TU DONG TIM DUONG DAN ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "raw_captures_goc")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "raw_captures")

def pause_exit():
    print("\n" + "-"*40)
    input("NHAN PHIM ENTER DE THOAT...")
    sys.exit()

# TIEN HANH IMPORT VOI CACH FIX CHO PYTHON 3.13
try:
    import cv2
    import numpy as np
    # Cach import moi de tranh loi 'solutions' tren 3.13
    import mediapipe as mp
    from mediapipe.python.solutions import face_detection as mp_face_detection
    USE_MEDIAPIPE = True
except Exception as e:
    print(f"[*] Canh bao: MediaPipe gap loi ({e}). Chuyen sang dung OpenCV Cascade (Dự phòng)...")
    USE_MEDIAPIPE = False

# CẤU HÌNH AI
TARGET_SIZE = 80
CROP_SCALE = 2.7  

def process_images():
    try:
        if not os.path.exists(INPUT_DIR):
            print(f"[!] LOI: Khong tim thay thu muc: {INPUT_DIR}")
            return
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        print(f"[*] Dang doc anh tu: {INPUT_DIR}")
        
        # Khoi tao bo phat hien
        if USE_MEDIAPIPE:
            print("[*] Su dung Mediapipe...")
            face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        else:
            print("[*] Su dung OpenCV Haarcascade...")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_detector = cv2.CascadeClassifier(cascade_path)

        files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg"))]
        if not files:
            print(f"[!] Canh bao: Thu muc anh goc dang rong.")
            return

        print(f"--- Bat dau xu ly {len(files)} anh ---")
        
        count = 0
        for filename in files:
            try:
                img_path = os.path.join(INPUT_DIR, filename)
                img = cv2.imread(img_path)
                if img is None: continue
                h, w, _ = img.shape
                
                faces = []
                if USE_MEDIAPIPE:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = face_detector.process(img_rgb)
                    if results.detections:
                        for detection in results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            faces.append((int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)))
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_GRAY if len(img.shape)==2 else cv2.COLOR_BGR2GRAY)
                    detected_faces = face_detector.detectMultiScale(gray, 1.1, 4)
                    for (x, y, fw, fh) in detected_faces:
                        faces.append((x, y, fw, fh))

                # Crop mat dau tien tim thay
                if faces:
                    x, y, fw, fh = faces[0]
                    cx, cy = x + fw // 2, y + fh // 2
                    side = int(max(fw, fh) * CROP_SCALE)
                    
                    # In thong tin debug cho 10 anh dau
                    if count < 10:
                        face_percent = (fw * fh) / (w * h) * 100
                        print(f"[*] File {filename}: Mat chiem {face_percent:.1f}% anh. Kick thuoc crop: {side}x{side}")

                    x1, y1 = cx - side // 2, cy - side // 2
                    x2, y2 = x1 + side, y1 + side

                    pad_x1 = max(0, -x1); pad_y1 = max(0, -y1)
                    pad_x2 = max(0, x2 - w); pad_y2 = max(0, y2 - h)

                    cropped = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    
                    if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
                        cropped = cv2.copyMakeBorder(cropped, pad_y1, pad_y2, pad_x1, pad_x2, 
                                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])

                    resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE))
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"crop_{filename}"), resized)
                    count += 1

                if count % 50 == 0 and count > 0:
                    print(f"[*] Da crop duoc {count} tam...")

            except Exception as e:
                pass # Bo qua anh loi nho

        print(f"\n--- XONG! Da tao ra {count} anh 80x80 trong '{OUTPUT_DIR}' ---")

    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    process_images()
    pause_exit()
