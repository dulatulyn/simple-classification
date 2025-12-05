import cv2
import numpy as np
import mediapipe as mp
import time
import sys
from predictor_service import PredictorService

mp_face_mesh = mp.solutions.face_mesh

def landmarks_to_array(face_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark], dtype=np.float32)

def check_camera_permission():
    print("Checking camera permissions...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera permission denied or not available")
        print("Please grant camera access in System Preferences > Security & Privacy > Camera")
        return False
    cap.release()
    return True

def run(camera_index=0):
    if not check_camera_permission():
        sys.exit(1)
    
    print("Opening camera...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Trying alternative camera backends...")
        for backend in [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2]:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                print(f"Camera opened with backend: {backend}")
                break
        else:
            raise RuntimeError("Camera not opened with any backend")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    ) as face_mesh:
        predictor = PredictorService()
        current_id = 0
        current_label = predictor.label_for_id(current_id)
        hamster_img = predictor.image_for_id(current_id)
        
        print("Camera ready. Press 'q' or ESC to quit.")
        
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                time.sleep(0.1)
                continue
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True
            
            if results.multi_face_landmarks:
                fl = results.multi_face_landmarks[0]
                coords = landmarks_to_array(fl)
                pred_id = predictor.predict_id(coords)
                if pred_id != current_id:
                    current_id = pred_id
                    current_label = predictor.label_for_id(pred_id)
                    hamster_img = predictor.image_for_id(pred_id)
            
            cv2.putText(
                frame, 
                f"{current_id} {current_label}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA
            )
            
            if hamster_img is not None:
                cv2.imshow("Hamster", hamster_img)
            
            cv2.imshow("Camera", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()