from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
import io
import os
import logging
import base64

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Inicializar FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta del modelo único
MODEL_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "ham10000_model.keras")

# Cargar modelo único
logging.info("Cargando modelo único...")
modelo = load_model(MODEL_PATH)
logging.info("Modelo cargado correctamente.")

# Formato de entrada (solo imagen ahora)
class InputData(BaseModel):
    image: str | None = None  # URL o base64

# Función auxiliar para procesar imagen
def process_image(img_input: str):
    try:
        if img_input.startswith("http"):
            response = requests.get(img_input)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            base64_data = img_input.split(",", 1)[1] if "," in img_input else img_input
            img_bytes = base64.b64decode(base64_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img = img.resize((224, 224))
        img_arr = np.array(img) / 255.0
        return np.expand_dims(img_arr, axis=0)

    except Exception as e:
        logging.error(f"Error al procesar la imagen: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {e}")

@app.post("/predict")
async def predict(data: InputData):
    try:
        logging.info(f"Datos recibidos: image={'presente' if data.image else None}")

        if not data.image:
            raise HTTPException(status_code=400, detail="No se envió ninguna imagen")

        # Procesar imagen
        logging.info("Procesando imagen...")
        img_arr = process_image(data.image)

        logging.info("Ejecutando predicción...")
        raw_pred = modelo.predict(img_arr)[0][0]

        riesgo = float(raw_pred * 99)

        resultado = {
            "riesgo": round(riesgo, 2)
        }

        logging.info(f"Resultado final: {resultado}")
        return resultado

    except Exception as e:
        logging.error(f"Error en /predict: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
