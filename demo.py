import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. TRAINING DATA (SUPERVISED)
# -----------------------------
# Features: [face_width, face_height, avg_pixel_intensity]
X_train = np.array([
    [100, 120, 180],  # Happy
    [105, 125, 175],  # Happy
    [95, 110, 170],   # Happy

    [90, 100, 130],   # Neutral
    [92, 105, 125],   # Neutral
    [88, 98, 120],    # Neutral

    [80, 90, 90],     # Sad
    [82, 92, 95],     # Sad
    [78, 88, 85]      # Sad
])

# Labels: 0=Happy, 1=Neutral, 2=Sad
y_train = np.array([
    0, 0, 0,
    1, 1, 1,
    2, 2, 2
])

# -----------------------------
# 2. TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

moods = ["Happy", "Neutral", "Sad"]

# -----------------------------
# 3. FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# 4. CAMERA SETUP
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# 5. GRAPH SETUP
# -----------------------------
plt.ion()
fig, ax = plt.subplots()

# -----------------------------
# 6. MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        avg_intensity = np.mean(face_roi)

        features = np.array([[w, h, avg_intensity]])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        mood_text = moods[prediction]

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Mood: {mood_text}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # -----------------------------
        # GRAPH UPDATE
        # -----------------------------
        ax.clear()
        ax.bar(moods, probabilities)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Mood Prediction Probability")
        plt.pause(0.01)

    cv2.imshow("Mood Prediction (Press Q to Quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# 7. CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
