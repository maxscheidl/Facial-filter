import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


class FaceDetector:

    def __init__(self, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.face_mesh = mp_face_mesh.FaceMesh(False, max_num_faces, refine_landmarks, min_detection_confidence,
                                               min_tracking_confidence)

    def detect_landmarks(self, image, draw=True):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if draw and results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, drawing_spec,
                                          mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec,
                                          mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_IRISES, drawing_spec,
                                          mp_drawing_styles.get_default_face_mesh_iris_connections_style())


        return results.multi_face_landmarks, image
