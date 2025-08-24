from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelos entrenados (.keras)
cnn_image_model = tf.keras.models.load_model("cnn_image_model.keras")
diameter_model = tf.keras.models.load_model("diameter_model.keras")
combined_model = tf.keras.models.load_model("combined_model.keras")

class InputData(BaseModel):
    image: str | None = None  # Acepta base64 o URL
    diametro: int | None = None


def preprocess_image(image_data):
    """Convierte la imagen en array normalizado para la CNN"""
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = img.resize((128, 128))  # mismo tamaño que usaste en el entrenamiento
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def clip_risk(value):
    """Limita el riesgo entre 1 y 99"""
    return float(np.clip(value, 1, 99))


@app.post("/predict")
async def predict(data: InputData):
    image_risk = None
    number_risk = None
    general_risk = None

    # --- Riesgo según imagen ---
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
            image_risk = clip_risk(cnn_image_model.predict(img_array)[0][0] * 100)  # Escalamos a porcentaje
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {e}")

    # --- Riesgo según diámetro ---
    try:
        if data.diametro is not None:
            diametro = np.array([[data.diametro]], dtype=np.float32)
            number_risk = clip_risk(diameter_model.predict(diametro)[0][0] * 100)
    except ValueError:
        raise HTTPException(status_code=400, detail="El diámetro debe ser un número entero válido")

    # --- Riesgo general (modelo combinado) ---
    if image_risk is not None and number_risk is not None:
        general_input = [np.array([[data.diametro]], dtype=np.float32), img_array]
        general_risk = clip_risk(combined_model.predict(general_input)[0][0] * 100)

    return {
        "riesgo_según_imagen": image_risk,
        "riesgo_según_diametro": number_risk,
        "riesgo_general": general_risk
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
