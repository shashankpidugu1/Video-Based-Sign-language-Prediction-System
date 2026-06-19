import os
import pickle

import mediapipe as mp
import cv2

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory where the dataset is stored
DATA_DIR = './data'

# Prepare containers for the data and labels
data = []
labels = []

# Loop through each class (A-Z)
for dir_ in os.listdir(DATA_DIR):
    if dir_.upper() in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):  # Ensure only A-Z classes are processed
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []

            x_ = []
            y_ = []

            # Read the image and convert to RGB
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image to find hands landmarks
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    # Normalize coordinates based on the minimum values
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))  # Normalize x coordinate
                        data_aux.append(y - min(y_))  # Normalize y coordinate

                # Add the processed data and labels
                data.append(data_aux)
                labels.append(dir_)

# Save the processed data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data processing completed and saved to 'data.pickle'.")
