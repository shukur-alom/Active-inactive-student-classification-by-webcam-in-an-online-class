import cv2
import time
import mediapipe as mp

y_val = 1 #attentive 1; inattentive 0
cam = cv2.VideoCapture("videos/attentive1.mp4")
file_data = open("data/attentive_2.csv", "w")

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

while True:
    _, frame = cam.read()
    frame = cv2.resize(frame, (900,550))
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
        for  landmark in RIGHT_EYE:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)

            value = (landmark.x * 100)*(landmark.y * 100) * 100

            print(value)
            file_data.write(f"{value}")
            file_data.write(",")
            
            cv2.circle(frame, (x, y), 1, (0, 255, 255))
            
        file_data.write(f"{y_val}")
        file_data.write("\n")
        #time.sleep(20)

    cv2.imshow('Eye', frame)
    cv2.waitKey(1)
