import cv2
import mediapipe as mp
import pickle


# open a file, where you stored the pickled data
file = open('models/model_random 2.pkl', 'rb')

data = pickle.load(file)

class_for_model = ["Inactive", "Active"]

cam = cv2.VideoCapture(0)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

while True:
    _, frame = cam.read()
    frame = cv2.resize(frame, (900, 550))
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        LEFT_EYE = [landmarks[362], landmarks[382], landmarks[381], landmarks[380], landmarks[374], landmarks[373], landmarks[390], landmarks[249],
                    landmarks[263], landmarks[466], landmarks[388], landmarks[387], landmarks[386], landmarks[385], landmarks[384], landmarks[398],
                    landmarks[474], landmarks[475], landmarks[476], landmarks[477]]

        for landmark in LEFT_EYE:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0))
        RIGHT_EYE = [landmarks[33], landmarks[7], landmarks[163], landmarks[144], landmarks[145], landmarks[153], landmarks[154], landmarks[155],
                     landmarks[133], landmarks[173], landmarks[157], landmarks[
                         158], landmarks[159], landmarks[160], landmarks[161], landmarks[246],
                     landmarks[469], landmarks[470], landmarks[471], landmarks[472]]
        raw_data = []
        for landmark in RIGHT_EYE:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            value = (landmark.x * 100)*(landmark.y * 100)*100
            raw_data.append(value)
            # print(value)
            cv2.circle(frame, (x, y), 1, (0, 255, 255))
        frame = cv2.putText(frame, f"{class_for_model[data.predict([raw_data])[0]]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Eye', frame)
    cv2.waitKey(1)
