import cv2
import mediapipe as mp # type: ignore
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((360, 640, 3), np.uint8)

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 20, 147)]
color_index = 0
thickness = 2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)

prev_x, prev_y = None, None

fist_detected = False

def is_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    if (index_tip.y < hand_landmarks.landmark[4].y and
        middle_tip.y < hand_landmarks.landmark[6].y and
        ring_tip.y < hand_landmarks.landmark[8].y and
        pinky_tip.y < hand_landmarks.landmark[10].y and
        thumb_tip.x < hand_landmarks.landmark[2].x):
        return True
    return False

def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if is_fist(hand_landmarks):
                if not fist_detected:
                    canvas = np.zeros((360, 640, 3), np.uint8)
                    fist_detected = True
            else:
                fist_detected = False
                
                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    lm_list.append((x, y))

                index_x, index_y = lm_list[8]
                thumb_x, thumb_y = lm_list[4]

                distance = calculate_distance((index_x, index_y), (thumb_x, thumb_y))

                thickness = int(min(max(distance / 5, 10), 40))

                if lm_list[8][1] < lm_list[6][1] and lm_list[12][1] < lm_list[10][1]:
                    prev_x, prev_y = None, None
                    
                    if index_y < 100:
                        if 50 < index_x < 150:
                            color_index = 0
                        elif 200 < index_x < 300:
                            color_index = 1
                        elif 350 < index_x < 450:
                            color_index = 2
                        elif 500 < index_x < 600:
                            color_index = 3

                elif lm_list[8][1] < lm_list[6][1]:
                    if prev_x is None or prev_y is None:
                        prev_x, prev_y = index_x, index_y

                    cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), colors[color_index], thickness)
                    prev_x, prev_y = index_x, index_y

    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    dark_canvas = cv2.addWeighted(canvas, 0.7, np.zeros_like(canvas), 0.3, 0)

    cv2.rectangle(frame, (50, 50), (150, 100), (0, 0, 255), -1)
    cv2.rectangle(frame, (200, 50), (300, 100), (0, 255, 0), -1)
    cv2.rectangle(frame, (350, 50), (450, 100), (255, 0, 0), -1)
    cv2.rectangle(frame, (500, 50), (600, 100), (255, 20, 147), -1)

    combined_frame = np.hstack((frame, dark_canvas))

    cv2.imshow("Webcam and Drawing Canvas", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
