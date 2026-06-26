import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.7
)

#Initializing the camera <-- 0 tells OpenCV to use default camera
cap = cv2.VideoCapture(0)

#Checking if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

#Infinite loop to continuously get frames from the camera  
while True:
    #Capture frame-by-frame; isScuccess = boolean if frame read correctly
    isSuccess, frame = cap.read()

    if not isSuccess:
        print("Error: Couldn't receive frame... Exiting program...")
        break

    #We need to convert to RGB cuz Mediapipe library was trained soley on RGB. 
    #Not BGR, Which is OpenCV's default color format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Processes frame and extracts landmarks
    result = hands.process(rgb_frame)

    #Flips camera feed horizontally to create a mirror effect
    frame = cv2.flip(frame, 1)

    if result.multi_hand_landmarks:
        #If looking at more than one hand, we'd need a for loop here instead of looking at js [0]
        hand_landmarks = result.multi_hand_landmarks[0]
        
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    #Display resulting frame in a window named "Camera Feed"
    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

#Deactivates camera & frees up resources when loop exited
#Otherwise, camera remains active and blocks other apps from using it
cap.release()

#Closes all OpenCV windows when program ends
cv2.destroyAllWindows()
    





    
