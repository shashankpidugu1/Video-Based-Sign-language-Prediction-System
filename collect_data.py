import os
import cv2
import string

# Directory to store the dataset
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the classes (A-Z)
classes = list(string.ascii_uppercase)
number_of_classes = len(classes)  # This will be 26 (A-Z)
dataset_size = 100  # Number of images per class

# Initialize camera (You can change the index if needed)
cap = cv2.VideoCapture(1)  # Change index if needed (0 or 1 or 2)

# Create subdirectories for each class
for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

# Collect data for each class
for j, class_name in enumerate(classes):
    print(f'Collecting data for class {class_name}')

    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame for {class_name}")
            break  # Exit if no frame is captured
        cv2.putText(frame, f'Ready for {class_name}? Press "Q" to start!',
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            done = True

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame for {class_name} at image {counter}")
            break  # Exit if no frame is captured
        print(f"Saving image {counter} for {class_name}")
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        image_path = os.path.join(DATA_DIR, class_name, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)
        print(f"Image saved at {image_path}")
        counter += 1

    print(f"Finished collecting data for class {class_name}")
    print("Moving on to the next class...")

cap.release()
cv2.destroyAllWindows()
