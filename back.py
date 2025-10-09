import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from collections import deque
import base64
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models globally
detector = HandDetector(maxHands=1)
classifier_letters = Classifier("Models/keras_model.h5", "Models/labels.txt")
classifier_extra = Classifier("Models/model/keras_model_extra.h5", "Models/model/labels_extra.txt")

labels_letters = [line.strip() for line in open("Models/labels.txt").readlines()]
labels_extra = [line.strip() for line in open("Models/model/labels_extra.txt").readlines()]

offset = 20
imgSize = 300

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    # Connection-specific state
    current_letter = ""
    current_word = ""
    sentence = ""
    prediction_buffer = deque(maxlen=10)
    frame_count = 0

    while True:
        try:
            message = await websocket.receive_text()
            data = json.loads(message)
            frame_count += 1
            print(f"\n[Frame {frame_count}] Received frame")

            if "frame" not in data:
                print("[Warning] No 'frame' key in data")
                continue

            # Decode base64 image from frontend
            img_data = data["frame"].split(",")[1]
            img_bytes = base64.b64decode(img_data)
            npimg = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if img is None:
                print("[Warning] Failed to decode image")
                continue

            # Detect hands
            hands, _ = detector.findHands(img)
            if not hands:
                print("[Info] No hands detected")
                await websocket.send_text(json.dumps({
                    "letter": current_letter,
                    "word": current_word,
                    "sentence": sentence,
                    "status": "no_hand"
                }))
                continue

            hand = hands[0]
            x, y, w, h = hand['bbox']
            print(f"[Info] Hand detected at x:{x}, y:{y}, w:{w}, h:{h}")

            # Preprocess hand image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            y1, y2 = max(0, y - offset), min(y + h + offset, img.shape[0])
            x1, x2 = max(0, x - offset), min(x + w + offset, img.shape[1])
            imgCrop = img[y1:y2, x1:x2]
            if imgCrop.size == 0:
                print("[Warning] Cropped image empty")
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
            except Exception as e:
                print(f"[Error] Image resize failed: {e}")
                continue

            # Predictions
            prediction1, index1 = classifier_letters.getPrediction(imgWhite, draw=False)
            prediction2, index2 = classifier_extra.getPrediction(imgWhite, draw=False)
            confidence1 = np.max(prediction1)
            confidence2 = np.max(prediction2)

            if confidence1 > confidence2:
                current_label = labels_letters[index1]
                confidence = confidence1
            else:
                current_label = labels_extra[index2]
                confidence = confidence2

            print(f"[Prediction] Raw: {current_label}, Confidence: {confidence:.2f}")

            if confidence > 0.85:
                prediction_buffer.append(current_label)

            if prediction_buffer:
                smoothed_label = max(set(prediction_buffer), key=prediction_buffer.count)
                print(f"[Prediction] Smoothed: {smoothed_label}")

                # Gesture handling
                # if smoothed_label == "RIGHT":
                #     if current_letter != "":
                #         current_word += current_letter
                #         print(f"[Action] RIGHT: Adding '{current_letter}' to current_word → {current_word}")
                #         current_letter = ""
                # elif smoothed_label == "SPACE":
                #     if current_word != "":
                #         sentence += current_word + " "
                #         print(f"[Action] SPACE: Adding '{current_word}' to sentence → {sentence}")
                #         current_word = ""
                # elif smoothed_label == "RESET":
                #     print("[Action] RESET: Clearing all")
                #     current_letter = ""
                #     current_word = ""
                #     sentence = ""
                # elif smoothed_label in ["HI", "MY"]:
                #     sentence += smoothed_label + " "
                #     print(f"[Action] Special word detected: {smoothed_label} → sentence: {sentence}")
                #     current_letter = ""
                # else:
                #     current_letter = smoothed_label
                #     print(f"[Update] current_letter set to {current_letter}")

            # Send response
            # result = {
            #     "letter": current_letter,  # e.g., "A", "B", "HI", "MY"
            #     "word": current_word,  # concatenated letters → words
            #     "sentence": sentence,  # concatenated words → sentence
            #     "status": "ok"
            # }
            # await websocket.send_text(json.dumps(result))
            #
            await websocket.send_text(json.dumps({"letter": current_letter}))
            await asyncio.sleep(0.05)

        except Exception as e:
            print(f"[Error] Exception: {e}")
            break

if __name__ == "__main__":
    print("[Server] Starting WebSocket server on ws://0.0.0.0:8000/ws")
    uvicorn.run(app, host="0.0.0.0", port=8000)
