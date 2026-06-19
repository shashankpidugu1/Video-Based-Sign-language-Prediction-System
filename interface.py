import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import threading

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Set up video capture
cap = cv2.VideoCapture(1)  # Use 1 for external camera or 0 for default camera

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels for A-Z
labels_dict = {i: chr(65 + i) for i in range(26)}

# Variables to store the sentence and the last predicted character
sentence = []
last_predicted_char = None

# Variables for stability check
prediction_history = []  # Stores recent predictions
stability_threshold = 10  # Number of consistent predictions required
cooldown_frames = 30  # Cooldown period after adding a character
cooldown_counter = 0  # Counter for cooldown

# Status variable
status = "Listening"

# Function to speak the sentence in a separate thread
def speak_sentence(sentence_text):
    global status
    try:
        # Use espeak to speak the sentence
        os.system(f'espeak "{sentence_text}"')
        print(f"Speaking: {sentence_text}")  # Debugging: Print the sentence to confirm
    except Exception as e:
        print(f"Error in TTS: {e}")
    finally:
        status = "Listening"  # Reset status after speaking

# Main loop
while True:
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    H, W, _ = frame.shape

    # Convert to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # Image to draw on
                hand_landmarks,  # Hand landmarks
                mp_hands.HAND_CONNECTIONS,  # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect landmarks for prediction
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Calculate bounding box for text placement
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Predict the sign language letter
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]  # Directly use the predicted label

            # Add the predicted character to the history
            prediction_history.append(predicted_character)

            # Keep only the last `stability_threshold` predictions
            if len(prediction_history) > stability_threshold:
                prediction_history.pop(0)

            # Check if the predictions are stable
            if len(prediction_history) == stability_threshold and all(c == predicted_character for c in prediction_history):
                if cooldown_counter == 0:  # Allow repeated characters
                    sentence.append(predicted_character)
                    last_predicted_char = predicted_character
                    cooldown_counter = cooldown_frames  # Start cooldown

            # Draw bounding box and predicted text on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    # Display the sentence on the screen with a background rectangle
    sentence_text = "Sentence: " + "".join(sentence)
    (text_width, text_height), _ = cv2.getTextSize(sentence_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    cv2.rectangle(frame, (10, 30), (20 + text_width, 70), (0, 0, 0), -1)  # Black background
    cv2.putText(frame, sentence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)  # White text

    # Display status with a background rectangle
    status_text = "Status: " + status
    (status_width, status_height), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    cv2.rectangle(frame, (10, H - 90), (20 + status_width, H - 30), (0, 0, 0), -1)  # Black background
    cv2.putText(frame, status_text, (10, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)  # White text

    # Display instructions with a background rectangle
    instruction_text = "Press 's' to speak, 'b' for backspace, 'c' to clear"
    (instruction_width, instruction_height), _ = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10, H - 30), (20 + instruction_width, H - 10), (0, 0, 0), -1)  # Black background
    cv2.putText(frame, instruction_text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # White text

    # Display the frame
    cv2.imshow('frame', frame)

    # Decrease cooldown counter
    if cooldown_counter > 0:
        cooldown_counter -= 1

    # Check for key presses
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):  # Quit the application
        break
    elif key & 0xFF == ord(' '):  # Spacebar to add a space between words
        sentence.append(' ')
    elif key & 0xFF == ord('s'):  # 's' key to speak the sentence
        sentence_text = "".join(sentence)
        if sentence_text.strip():  # Check if the sentence is not empty
            status = "Speaking"
            threading.Thread(target=speak_sentence, args=(sentence_text,)).start()  # Start TTS in a new thread
    elif key & 0xFF == ord('b'):  # 'b' key for backspace
        if sentence:  # Check if the sentence is not empty
            sentence.pop()  # Remove the last character
    elif key & 0xFF == ord('c'):  # 'c' key to clear the sentence
        sentence = []

# Release resources and close the window
cap.release()
cv2.destroyAllWindows()