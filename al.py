import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the landmarks for the index, middle, and thumb fingers
                index_lm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_lm = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_lm = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Calculate the Euclidean distance between the index and middle finger landmarks
                im_distance = math.sqrt((index_lm.x - middle_lm.x)**2 + (index_lm.y - middle_lm.y)**2 + (index_lm.z - middle_lm.z)**2)
                print(f"Distance between index and middle fingers: {im_distance:.2f}")

                # Calculate the Euclidean distance between the index and thumb finger landmarks
                it_distance = math.sqrt((index_lm.x - thumb_lm.x)**2 + (index_lm.y - thumb_lm.y)**2 + (index_lm.z - thumb_lm.z)**2)
                print(f"Distance between index and thumb fingers: {it_distance:.2f}")
 # Render the landmarks and annotations on the image
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('ai virtual mouse', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break 

cap.release()
cv2.destroyAllWindows()