import cv2
import numpy as np
import math
import base64
import json
import asyncio
from collections import deque
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Models/keras_model.h5", "Models/labels.txt")

offset = 20
imgSize = 300
labels = [chr(i) for i in range(65, 91)]  # A–Z

# For smoother predictions
prediction_buffer = deque(maxlen=10)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()
            frame_data = json.loads(data)["frame"]

            # Decode base64 image from frontend
            frame_bytes = base64.b64decode(frame_data.split(",")[1])
            np_img = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            hands, img = detector.findHands(img)

            current_letter = ""

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

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

                if confidence > 0.85:
                    prediction_buffer.append(labels[index])

                # Smooth by majority vote
                if len(prediction_buffer) > 0:
                    current_letter = max(set(prediction_buffer), key=prediction_buffer.count)

            # Send letter back
            await websocket.send_text(json.dumps({"letter": current_letter}))
            await asyncio.sleep(0.05)

    except Exception as e:
        print("⚠️ WebSocket closed:", e)
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
