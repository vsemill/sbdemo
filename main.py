from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

app = FastAPI()

# Enable CORS so Netlify can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the Model (Using Nano 'n' version for Render Free Tier)
print("Loading Brain...")
model = YOLO('yolov8n-worldv2.pt')

# 2. Your Kerala Dictionary
model.set_classes([
    "fresh jackfruit", "rotten jackfruit",
    "fresh tapioca root", "rotten tapioca",
    "fresh coconut", "rotten coconut",
    "fresh drumstick vegetable", "rotten drumstick",
    "fresh snake gourd", "rotten snake gourd",
    "fresh bitter gourd", "rotten bitter gourd",
    "fresh ash gourd", "rotten ash gourd",
    "fresh yam", "rotten yam",
    "fresh lady finger", "rotten lady finger",
    "green plantain", "rotten plantain",
    "yellow plantain", "black spots banana",
    "small banana",
    "green chili", "rotten chili",
    "ginger root", "rotten ginger",
    "garlic bulb", "rotten garlic",
    "red onion", "rotten onion",
    "fresh mango", "rotten mango",
    "fresh papaya", "rotten papaya",
    "fresh pineapple", "rotten pineapple",
    "mold patch", "fungus spot", "soft rotten spot"
])

@app.get("/")
def home():
    return {"status": "Brain is Awake! 🧠"}

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    # Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Predict
    results = model.predict(img, conf=0.15)

    # Plot & Return
    for r in results:
        im_plot = r.plot()
        # Encode directly to JPEG
        _, encoded_img = cv2.imencode('.jpg', im_plot)
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")