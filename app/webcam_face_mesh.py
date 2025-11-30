import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def landmarks_to_array(face_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark], dtype=np.float32)

def run(camera_index = 0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened")
    with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True
            if results.multi_face_landmarks:
                for fl in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        fl,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec = None,
                        connection_drawing_spec = mp_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        frame,
                        fl,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec = None,
                        connection_drawing_spec = mp_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        frame,
                        fl,
                        mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec = None,
                        connection_drawing_spec = mp_styles.get_default_face_mesh_iris_connections_style(),
                    )
            cv2.imshow("Face Mesh", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()