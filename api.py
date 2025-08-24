from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf
import os
import zipfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Función para descargar y descomprimir modelos desde Google Drive ---
def download_model_gdrive(file_id, zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"Descargando {extract_to} desde Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(url)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
        print(f"{extract_to} listo.")

# --- IDs de los archivos en Google Drive ---
download_model_gdrive("1vWyH3IJXVe9vV-XgtZn0Rbt3YrGxsOE6", "modelo_imagen.zip", "modelo_imagen.keras")
download_model_gdrive("1v3UjiUkyTCUiZuKopGm113DMaEuGyr2a", "modelo_diametro.zip", "modelo_diametro.keras")
download_model_gdrive("1nhZORXhSgCo06xIR7ZmAUrmBHVkf08oS", "modelo_dermascan.zip", "modelo_dermascan.keras")

# --- Cargar modelos con nombres correctos ---
cnn_image_model = tf.keras.models.load_model("modelo_imagen.keras")
diameter_model = tf.keras.models.load_model("modelo_diametro.keras")
combined_model = tf.keras.models.load_model("modelo_dermascan.keras")

# --- Clase para recibir los datos ---
class InputData(BaseModel):
    image: str | None = None  # base64 o URL
    diametro: int | None = None

# --- Preprocesamiento de imagen ---
def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Limitar riesgo entre 1 y 99 ---
def clip_risk(value):
    return float(np.clip(value, 1, 99))

# --- Endpoint de predicción ---
@app.post("/predict")
async def predict(data: InputData):
    image_risk = None
    number_risk = None
    general_risk = None

    # Riesgo según imagen
    if data.image:
        try:
            if data.image.startswith("http"):
                response = requests.get(data.image)
                response.raise_for_status()
                image_data = response.content
            else:
                base64_data = data.image.split(",", 1)[1] if "," in data.image else data.image
                image_data = base64.b64decode(base64_data)

            img_array = preprocess_image(image_data)
            image_risk = clip_risk(cnn_image_model.predict(img_array)[0][0] * 100)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {e}")

    # Riesgo según diámetro
    if data.diametro is not None:
        try:
            diametro_array = np.array([[data.diametro]], dtype=np.float32)
            number_risk = clip_risk(diameter_model.predict(diametro_array)[0][0] * 100)
        except ValueError:
            raise HTTPException(status_code=400, detail="El diámetro debe ser un número entero válido")

    # Riesgo combinado
    if image_risk is not None and number_risk is not None:
        try:
            general_input = [diametro_array, img_array]
            general_risk = clip_risk(combined_model.predict(general_input)[0][0] * 100)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error en predicción combinada: {e}")

    return {
        "riesgo_según_imagen": image_risk,
        "riesgo_según_diametro": number_risk,
        "riesgo_general": general_risk
    }

# --- Ejecutar API ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
