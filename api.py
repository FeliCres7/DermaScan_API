from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import requests
from PIL import Image
import io
import random

# pip install fastapi uvicorn requests Pillow


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    image_url: HttpUrl | None = None
    number: int | None = None

@app.post("/predict")
async def predict(data: InputData):
    image_risk = None
    number_risk = None
    general_risk = None

    if data.image_url:
        try:
            response = requests.get(data.image_url)
            response.raise_for_status()
            Image.open(io.BytesIO(response.content))  # validar imagen
            image_risk = random.randint(0, 99)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {e}")

    if data.number is not None:
        number_risk = random.randint(0, 99)

    if image_risk is not None and number_risk is not None:
        general_risk = random.randint(0, 99)

    return {
        "riesgo_según_imagen": image_risk,
        "riesgo_según_diametro": number_risk,
        "riesgo_general": general_risk
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
