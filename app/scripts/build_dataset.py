import os
import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR=str(ROOT / "docs/media/img/me")
OUTPUT_DIR=str(ROOT / "docs/media/dataset")
IMAGE_EXTS={".jpg", ".jpeg", ".png", ".bmp"}

mp_face_mesh = mp.solutions.face_mesh

def dir_is_empty(path):
    try:
        return not any(os.scandir(path))
    except FileNotFoundError:
        return True
    
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_coords(image_bgr, face_mesh):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None
    fl = results.multi_face_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in fl.landmark], dtype=np.float32)

def main():
    ensure_dir(OUTPUT_DIR)
    if not dir_is_empty(OUTPUT_DIR):
        print("Dataset folder is not empty. No files written")
        return
    if not os.path.isdir(INPUT_DIR):
        print("Input directory not found: ", INPUT_DIR)
        sys.exit(1)
    total = 0
    saved = 0

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        for root, _, files in os.walk(INPUT_DIR):
            label = os.path.basename(root)

            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue
                total += 1
                img_path = os.path.join(root, f)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                coords = extract_coords(img, face_mesh)
                if coords is None:
                    continue

                base = os.path.splitext(f)[0]
                out_name = f"{label}_{base}.npy"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                np.save(out_path, coords)
                saved += 1
                print(out_name)
    print(f"Processed: {total}, Saved: {saved}")

if __name__ == "__main__":
    main()
