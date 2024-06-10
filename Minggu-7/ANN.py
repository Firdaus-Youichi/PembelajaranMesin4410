import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)

# One hot encode the target values
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = Sequential()

# Add input layer and first hidden layer
model.add(Dense(units=10, activation='relu', input_shape=(X_train.shape[1],)))

# Add second hidden layer
model.add(Dense(units=10, activation='relu'))

# Add output layer
model.add(Dense(units=y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# If you want to see the actual predictions versus the true values
true_classes = np.argmax(y_test, axis=1)
print(f'Predicted classes: {predicted_classes}')
print(f'True classes: {true_classes}')

