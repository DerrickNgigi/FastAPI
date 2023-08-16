import cv2
import numpy as np
from keras.models import load_model
from fastapi import FastAPI, Form

app = FastAPI()

# Load the pre-trained model
model = load_model("model/Final_weights.h5")

# Define labels and other parameters
labels = {0: "Neutral", 1: "Porn", 2: "Sexy"}
# Supported file extensions for videos and images
supported_video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
supported_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    return frame

def classify_image(image):
    preds = model.predict(np.expand_dims(image, axis=0))[0]
    i = np.argmax(preds)
    return labels[i]

@app.post("/explicit/")
async def classify_content(url: str):
    print(f"Received URL: {url}")

    input_path = url  # Replace with the path to your input file

    # Check if the input path is a supported format
    if not any(extension in input_path.lower() for extension in supported_video_extensions + supported_image_extensions):
        print("Unsupported file format.")
        sys.exit(1)

    # Check if the input path is a video or image
    is_video = any(extension in input_path.lower() for extension in supported_video_extensions)
    is_image = any(extension in input_path.lower() for extension in supported_image_extensions) and not is_video

    if is_video:
        vs = cv2.VideoCapture(input_path)
    else:
        img = cv2.imread(input_path)

    label_counts = {"Neutral": 0, "Porn": 0, "Sexy": 0}

    while True:
        if is_video:
            grabbed, frame = vs.read()
            if not grabbed:
                break
        else:
            frame = img.copy()

        if is_video:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        (H, W) = frame.shape[:2]
        processed_frame = process_frame(frame)

        preds = model.predict(np.expand_dims(processed_frame, axis=0))[0]
        i = np.argmax(preds)
        label = labels[i]
        label_counts[label] += 1
        text = f"activity: {label}:"
        print(text)

        if label == "Porn":
            print("[INFO] Porn content detected.")
            return "Porn"

    print("[INFO] cleaning up...")
    if is_video:
        vs.release()

    if label_counts["Porn"] > 0:
        print("[INFO] Porn content detected.")
        return "Porn"
    elif label_counts["Sexy"] > label_counts["Neutral"]:
        print("[INFO] Content classified as Sexy.")
        return "Sexy"
    else:
        print("[INFO] Content classified as Neutral.")
        return "Neutral"
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)