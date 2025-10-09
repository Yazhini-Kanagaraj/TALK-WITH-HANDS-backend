import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from collections import deque

# Initialize
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load both models
classifier_letters = Classifier("Models/keras_model.h5", "Models/labels.txt")
classifier_extra = Classifier("Models/model/keras_model_extra.h5", "Models/model/labels_extra.txt")

# Load labels
labels_letters = [line.strip() for line in open("Models/labels.txt").readlines()]
labels_extra = [line.strip() for line in open("Models/model/labels_extra.txt").readlines()]

offset = 20
imgSize = 300

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

        # Predictions from both models
        prediction1, index1 = classifier_letters.getPrediction(imgWhite, draw=False)
        prediction2, index2 = classifier_extra.getPrediction(imgWhite, draw=False)

        confidence1 = np.max(prediction1)
        confidence2 = np.max(prediction2)

        # Choose the higher-confidence model
        if confidence1 > confidence2:
            current_label = labels_letters[index1]
            confidence = confidence1
        else:
            current_label = labels_extra[index2]
            confidence = confidence2

        # Stability threshold
        if confidence > 0.85:
            prediction_buffer.append(current_label)

        # Smooth prediction
        if len(prediction_buffer) > 0:
            smoothed_label = max(set(prediction_buffer), key=prediction_buffer.count)
            cv2.putText(imgOutput, f"Prediction: {smoothed_label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
            cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Handle gestures
            if smoothed_label == "RIGHT":
                if current_letter != "":
                    current_word += current_letter
                    current_letter = ""
            elif smoothed_label == "SPACE":
                if current_word != "":
                    sentence += current_word + " "
                    current_word = ""
            elif smoothed_label in ["HI", "MY"]:
                sentence += smoothed_label + " "
                current_letter = ""
            else:
                current_letter = smoothed_label

    # Display info
    cv2.putText(imgOutput, f"Letter: {current_letter}", (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(imgOutput, f"Word: {current_word}", (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(imgOutput, f"Sentence: {sentence}", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(imgOutput, "Gestures: RIGHT=Add Letter  SPACE=Add Word  Q=Quit",
                (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    cv2.imshow("Sign-to-Text", imgOutput)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
