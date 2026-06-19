import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract the data and labels
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# If labels are not in the correct A-Z format, we map them to 'A' to 'Z'
# Assumes that labels are already from A-Z, but if the labels are numeric or different, you can adjust accordingly
# Ensure that your dataset only contains labels from 'A' to 'Z' (excluding '0' to '9' for this task)

# Filter the labels and data to only include A-Z
valid_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
valid_indices = [i for i, label in enumerate(labels) if label in valid_labels]

# Filter the data and labels accordingly
data = data[valid_indices]
labels = labels[valid_indices]

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model with the training data
model.fit(x_train, y_train)

# Predict the labels of the test set
y_predict = model.predict(x_test)

# Calculate and print the accuracy score
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model using pickle
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()