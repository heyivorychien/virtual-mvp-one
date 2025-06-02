import cv2
import mediapipe as mp
import json

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Store results in a list
landmark_data = []

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
                )

                # Extract normalized coordinates
                face_coords = []
                for pt in landmarks.landmark:
                    face_coords.append({
                        "x": pt.x,
                        "y": pt.y,
                        "z": pt.z
                    })
                landmark_data = face_coords  # store only last frame for now

        cv2.imshow("Face Detection", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

# Save to JSON file
with open("landmarks.json", "w") as f:
    json.dump(landmark_data, f, indent=2)

print("✅ Landmarks saved to landmarks.json")