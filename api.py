from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import requests
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Inicializar app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelos desde el repo (mismo directorio que api.py)
model_img = load_model("modelo_imagen.keras")
model_diam = load_model("modelo_diametro.keras")
modelo_general = load_model("modelo_dermascan.keras")

# Inicializar scaler para diámetro (asumimos rango 0-1, si querés precisión podes guardarlo previamente)
scaler = MinMaxScaler()
scaler.fit(np.array([[0], [50]]))  # Ajustalo al rango de tus diámetros

# Formato de entrada
class InputData(BaseModel):
    image_url: HttpUrl | None = None
    number: int | None = None

@app.post("/predict")
async def predict(data: InputData):
    image_risk = None
    number_risk = None
    general_risk = None

    try:
        if data.image_url:
            response = requests.get(str(data.image_url))
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            img = img.resize((224, 224))
            img_arr = np.array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)

            # Predicción solo imagen
            image_risk = float(model_img.predict(img_arr)[0][0])

        if data.number is not None:
            num_scaled = scaler.transform(np.array([[data.number]]))
            number_risk = float(model_diam.predict(num_scaled)[0][0])

        if data.image_url and data.number is not None:
            general_risk = float(modelo_general.predict([img_arr, num_scaled])[0][0])

        return {
            "riesgo_según_imagen": image_risk,
            "riesgo_según_diametro": number_risk,
            "riesgo_general": general_risk
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
