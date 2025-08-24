from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
import numpy as np
from PIL import Image
import requests
import io
import os
from sklearn.preprocessing import MinMaxScaler

# Inicializar FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas relativas para Render
MODEL_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_IMG_PATH = os.path.join(MODEL_DIR, "modelo_imagen.keras")
MODEL_DIAM_PATH = os.path.join(MODEL_DIR, "modelo_diametro.keras")
MODEL_GENERAL_PATH = os.path.join(MODEL_DIR, "modelo_dermascan.keras")

# Cargar modelos
model_img = load_model(MODEL_IMG_PATH)
model_diam = load_model(MODEL_DIAM_PATH)
modelo_general = load_model(MODEL_GENERAL_PATH)

scaler = MinMaxScaler()
min_diameter, max_diameter = 0.5, 50.0  # <--- ajusta según tu dataset
scaler.min_ = np.array([min_diameter])
scaler.scale_ = np.array([max_diameter - min_diameter])

# Formato de entrada
class InputData(BaseModel):
    image_url: HttpUrl | None = None
    number: float | None = None

# Función auxiliar para cargar imagen
def load_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    img = img.resize((224, 224))
    img_arr = np.array(img) / 255.0
    return np.expand_dims(img_arr, axis=0)

@app.post("/predict")
async def predict(data: InputData):
    try:
        image_risk = None
        number_risk = None
        general_risk = None

        if data.image_url:
            img_arr = load_image_from_url(str(data.image_url))
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

# Para desarrollo local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
