import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

data_dir = "dataset"
X, y = [], []

for gesture in os.listdir(data_dir):
    gesture_dir = os.path.join(data_dir, gesture)
    for file in os.listdir(gesture_dir):
        with open(os.path.join(gesture_dir, file), "r") as f:
            landmarks = list(map(float, f.read().split(",")))
            X.append(landmarks)
            y.append(gesture)

X = np.array(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

model.save("gesture_model.h5")

# Save label encoder
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model and label encoder saved.")
