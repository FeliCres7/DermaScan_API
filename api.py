from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from PIL import Image
import io
import random
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    image: str | None = None  # Acepta base64 o URL
    diametro: int | None = None

@app.post("/predict")
async def predict(data: InputData):
    image_risk = None
    number_risk = None
    general_risk = None

    if data.image:
        try:
            if data.image.startswith("http"):
                # URL: descargar imagen
                response = requests.get(data.image)
                response.raise_for_status()
                image_data = response.content
            else:
                # Base64: limpiar prefijo si lo tiene
                base64_data = data.image.split(",", 1)[1] if "," in data.image else data.image
                image_data = base64.b64decode(base64_data)

            Image.open(io.BytesIO(image_data))  # Validar imagen
            image_risk = random.randint(0, 99)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {e}")

    if data.diametro is not None:
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
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
