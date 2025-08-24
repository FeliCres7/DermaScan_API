from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
import io
import os
import logging

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

# Rutas relativas para Render
MODEL_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_IMG_PATH = os.path.join(MODEL_DIR, "modelo_imagen.keras")
MODEL_DIAM_PATH = os.path.join(MODEL_DIR, "modelo_diametro.keras")
MODEL_GENERAL_PATH = os.path.join(MODEL_DIR, "modelo_dermascan.keras")

# Cargar modelos
logging.info("Cargando modelos...")
model_img = load_model(MODEL_IMG_PATH)
model_diam = load_model(MODEL_DIAM_PATH)
modelo_general = load_model(MODEL_GENERAL_PATH)
logging.info("Modelos cargados correctamente.")

# Rango de diámetros en tu dataset
min_diameter, max_diameter = 0.5, 50.0  

# Formato de entrada
class InputData(BaseModel):
    fotos: HttpUrl | None = None
    diametro: float | None = None

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
        img_arr = None
        num_scaled = None

        logging.info(f"Datos recibidos: fotos={data.fotos}, diametro={data.diametro}")

        # Procesamiento de la imagen
        if data.fotos:
            logging.info("Procesando imagen desde URL...")
            response = requests.get(data.fotos)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image = image.resize((224, 224))
            img_arr = np.expand_dims(np.array(image) / 255.0, axis=0)

            raw_img_pred = model_img.predict(img_arr)[0][0]
            logging.info(f"Predicción cruda imagen: {raw_img_pred}")
            pred_img = float(raw_img_pred * 99)

        # Procesamiento del número (diámetro)
        if data.diametro is not None:
            logging.info(f"Escalando número {data.diametro} con rango ({min_diameter}, {max_diameter})")
            num_scaled = np.array([[(data.diametro - min_diameter) / (max_diameter - min_diameter)]])
            raw_diam_pred = model_diam.predict(num_scaled)[0][0]
            logging.info(f"Predicción cruda diámetro: {raw_diam_pred}")
            pred_diam = float(raw_diam_pred * 99)

        # Predicción combinada
        if data.fotos and data.diametro is not None:
            logging.info("Ejecutando predicción general con imagen + número...")
            raw_gen_pred = modelo_general.predict([img_arr, num_scaled])[0][0]
            logging.info(f"Predicción cruda general: {raw_gen_pred}")
            pred_general = float(raw_gen_pred * 99)
        else:
            pred_general = None

        resultado = {
            "riesgo_según_imagen": round(pred_img, 2) if data.fotos else None,
            "riesgo_según_diametro": round(pred_diam, 2) if data.diametro is not None else None,
            "riesgo_general": round(pred_general, 2) if pred_general is not None else None,
        }

        logging.info(f"Resultado final: {resultado}")
        return resultado

    except Exception as e:
        logging.error(f"Error en /predict: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Para desarrollo local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
