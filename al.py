import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils

class HandRecog:
    def __init__(self, label):
        self.label = label
        self.hand_result = None

    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_dist(self, point):
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        dist = math.sqrt(dist)
        return dist

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize HandRecog object
hand_recog = HandRecog("MAJOR")

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a single frame from the video capture
    ret, frame = cap.read()

    # Flip the frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands solution
    results = hands.process(image)

    # Update the HandRecog object with the new hand result
    if results.multi_hand_landmarks:
        hand_recog.update_hand_result(results.multi_hand_landmarks[0])

        # Get the distance between the index and middle finger landmarks
        index_finger = 8
        middle_finger = 12
        distance = hand_recog.get_dist([index_finger, middle_finger])
        print(distance)

    # Render the landmarks and annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

    # Convert the RGB image back to BGR for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display the image with OpenCV
    cv2.imshow('ai virtual mouse', image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

