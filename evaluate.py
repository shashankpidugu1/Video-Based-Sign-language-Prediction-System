import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract the data and labels
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Ensure only A-Z labels are used
valid_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
valid_indices = [i for i, label in enumerate(labels) if label in valid_labels]

data = data[valid_indices]
labels = labels[valid_indices]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict the labels of the test set
y_predict = model.predict(x_test)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_predict)
print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

# Generate and print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_predict))

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=valid_labels, yticklabels=valid_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Cross-validation scores
cv_scores = cross_val_score(model, data, labels, cv=5)
print(f"Cross-validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Save the trained model only if it doesn't already exist