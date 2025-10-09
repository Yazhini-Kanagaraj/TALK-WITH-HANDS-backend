import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from collections import deque

# Initialize
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Models/keras_model.h5", "Models/labels.txt")

offset = 20
imgSize = 300

# Updated labels (your retrained model)
# labels = [
#     "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
#     "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
#     "HI", "MY", "RIGHT", "SPACE"
# ]

labels = ["A","B","C","D","E","F","G"," H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

# State variables
prediction_buffer = deque(maxlen=10)
current_letter = ""
current_word = ""
sentence = ""

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crop and preprocess image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        y1, y2 = max(0, y - offset), min(y + h + offset, img.shape[0])
        x1, x2 = max(0, x - offset), min(x + w + offset, img.shape[1])
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w
        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
        except:
            continue

        # Predict gesture
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        confidence = np.max(prediction)

        if confidence > 0.85:  # Slightly stricter for better stability
            prediction_buffer.append(labels[index])

        # Smooth prediction using majority vote
        if len(prediction_buffer) > 0:
            current_label = max(set(prediction_buffer), key=prediction_buffer.count)
            cv2.putText(imgOutput, f"Prediction: {current_label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
            cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Handle special gestures
            if current_label == "RIGHT":
                # Finalize letter or word
                if current_letter != "":
                    current_word += current_letter
                    current_letter = ""
            elif current_label == "SPACE":
                # Add current word to sentence
                if current_word != "":
                    sentence += current_word + " "
                    current_word = ""
            elif current_label in ["HI", "MY"]:
                # Directly add known word gestures
                sentence += current_label + " "
                current_letter = ""
            else:
                # Store current detected letter (A–Z)
                current_letter = current_label

    # Display text info
    cv2.putText(imgOutput, f"Letter: {current_letter}", (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(imgOutput, f"Word: {current_word}", (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(imgOutput, f"Sentence: {sentence}", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(imgOutput, "Gestures: RIGHT=Add Letter  SPACE=Add Word  Q=Quit",
                (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.imshow("Sign-to-Text", imgOutput)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
