import os
import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import zipfile
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# STEP 1: DOWNLOAD DATASET
# ----------------------------
url = "https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/master.zip"
zip_path = "dataset.zip"

if not os.path.exists("dataset"):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

    os.rename("Sign-Language-Digits-Dataset-master", "dataset")

print("Dataset Ready ✅")

dataset_path = "dataset/Dataset"

# ----------------------------
# STEP 2: MEDIAPIPE SETUP
# ----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)
mp_draw = mp.solutions.drawing_utils

# ----------------------------
# STEP 3: EXTRACT LANDMARKS
# ----------------------------
def extract_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)
        return landmarks
    return None

# ----------------------------
# STEP 4: LOAD DATASET
# ----------------------------
data, labels = [], []

for label, folder in enumerate(os.listdir(dataset_path)):
    path = os.path.join(dataset_path, folder)

    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name))

        if img is None:
            continue

        lm = extract_landmarks(img)

        if lm:
            data.append(lm)
            labels.append(label)

X = np.array(data)
y = np.array(labels)

print("Dataset Loaded:", X.shape)

# ----------------------------
# STEP 5: TRAIN MODEL
# ----------------------------
model = RandomForestClassifier()
model.fit(X, y)

print("Model Trained ✅")

# ----------------------------
# STEP 6: REAL-TIME WEBCAM
# ----------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            prediction = model.predict([landmarks])[0]

            cv2.putText(frame, str(prediction), (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()