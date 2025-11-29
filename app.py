import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import numpy as np


"""
Simple Paddy Disease Prediction API for Railway

- Exposes:
    GET  /          -> health check
    POST /predict   -> image upload (form-data "image") + prediction

- Model loading:
    - Supports Keras (.h5, .keras) via TensorFlow / tf.keras
    - Supports ONNX (.onnx) via onnxruntime
    - Default model path: ./model/model.h5  (relative to this file)
    - You can override via MODEL_PATH environment variable on Railway.
"""


app = Flask(__name__)
CORS(app)

# ---- Model loading ----

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "model.keras")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

model = None
model_type = None  # "keras" or "onnx"


def load_model():
    """
    Load the ML model depending on file extension.
    - .h5 / .keras  -> Keras (TensorFlow)
    - .onnx         -> ONNXRuntime
    """
    global model, model_type

    if model is not None:
        return model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    ext = os.path.splitext(MODEL_PATH)[1].lower()

    if ext in [".h5", ".keras"]:
        # Keras / TensorFlow model
        import tensorflow as tf  # noqa: F401  (ensures TF is available)
        from tensorflow import keras

        model = keras.models.load_model(MODEL_PATH)
        model_type = "keras"
        print(f"✅ Loaded Keras model from: {MODEL_PATH}")

    elif ext == ".onnx":
        import onnxruntime as ort

        model = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        model_type = "onnx"
        print(f"✅ Loaded ONNX model from: {MODEL_PATH}")

    else:
        raise ValueError(
            f"Unsupported model format '{ext}'. "
            "Use .h5 / .keras (Keras) or .onnx (ONNX)."
        )

    return model


def preprocess_image(file_storage, target_size=(224, 224)):
    """
    Convert uploaded image to a normalized numpy batch.
    - Resize to target_size (default 224x224)
    - Normalize to [0, 1]
    - Add batch dimension -> shape (1, H, W, 3)
    """
    image = Image.open(file_storage).convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image).astype("float32") / 255.0
    # (H, W, 3) -> (1, H, W, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr


# Disease classes (10-class model)
DISEASE_CLASSES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "status": "ok",
            "message": "Paddy Disease Prediction API is running",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    # Check file
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (form-data field 'image')"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Load model (lazy)
        mdl = load_model()

        # Preprocess
        input_data = preprocess_image(image_file)  # shape (1, H, W, 3)

        # Predict based on model type
        if model_type == "keras":
            preds = mdl.predict(input_data, verbose=0)[0]  # shape (num_classes,)
        elif model_type == "onnx":
            input_name = mdl.get_inputs()[0].name
            preds = mdl.run(None, {input_name: input_data})[0][0]
        else:
            return jsonify({"error": "Model type not initialized"}), 500

        preds = np.array(preds).astype("float32")

        # Softmax for probabilities if not already
        if preds.ndim == 1:
            exp = np.exp(preds - np.max(preds))
            probs = exp / exp.sum()
        else:
            probs = preds

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])

        # Map index to label (fallback if lengths differ)
        if top_idx < len(DISEASE_CLASSES):
            label = DISEASE_CLASSES[top_idx]
        else:
            label = f"class_{top_idx}"

        all_predictions = {
            (DISEASE_CLASSES[i] if i < len(DISEASE_CLASSES) else f"class_{i}"): float(
                probs[i]
            )
            for i in range(len(probs))
        }

        return jsonify(
            {
                "prediction": label,
                "confidence": confidence,
                "all_predictions": all_predictions,
            }
        )

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    # Railway will usually use gunicorn, but this is handy for local testing.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


