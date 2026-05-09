import base64
import os
import time
from io import BytesIO

import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from pyngrok import ngrok

app = Flask(__name__)

THAI_DIGIT_CLASSES = [11, 12, 13, 14, 15]
MODEL_PATH = 'thai_digit_model.h5'
_model = None


def load_prediction_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')

    if os.path.getsize(MODEL_PATH) == 0:
        raise ValueError(f'Model file is empty: {MODEL_PATH}')

    from keras.models import load_model

    _model = load_model(MODEL_PATH, compile=False)
    return _model


def preprocess_prediction_image(img_b64):
    if ',' in img_b64:
        img_b64 = img_b64.split(',', 1)[1]

    image_bytes = base64.b64decode(img_b64)
    image = Image.open(BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=(0, -1))
    return array


@app.route('/')
def index():
    """หน้าแรก -> หน้ารวบรวมข้อมูล (เหมือนเดิม)"""
    return render_template('index.html')


@app.route('/predict-page')
def predict_page():
    """หน้า prediction แยกอีก URL"""
    return render_template('predict.html')


@app.route('/save', methods=['POST'])
def save():
    data = request.get_json(silent=True)
    if not data or 'label' not in data or 'image' not in data:
        return jsonify({'error': 'Invalid data'}), 400

    label = str(data['label'])
    img_b64 = data['image']

    folder = os.path.join('dataset', label)
    os.makedirs(folder, exist_ok=True)

    timestamp = int(time.time() * 1000)
    filename = f"{label}_{timestamp}.png"
    filepath = os.path.join(folder, filename)

    with open(filepath, 'wb') as f:
        f.write(base64.b64decode(img_b64.split(',', 1)[-1]))

    return jsonify({'status': 'success', 'filename': filename})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'Invalid data'}), 400

    try:
        model = load_prediction_model()
        input_tensor = preprocess_prediction_image(data['image'])
        predictions = model.predict(input_tensor, verbose=0)
        scores = np.asarray(predictions[0], dtype=np.float32)
        predicted_index = int(np.argmax(scores))
        confidence_score = round(float(scores[predicted_index]) * 100.0, 2)

        if scores.shape[-1] == len(THAI_DIGIT_CLASSES):
            thai_digit = THAI_DIGIT_CLASSES[predicted_index]
        else:
            thai_digit = predicted_index

        return jsonify({
            'predicted_class': predicted_index,
            'thai_digit': thai_digit,
            'confidence_score': confidence_score,
        })
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({'error': f'Prediction unavailable: {exc}'}), 503
    except Exception as exc:
        return jsonify({'error': f'Prediction failed: {exc}'}), 500


if __name__ == '__main__':
    os.makedirs('dataset', exist_ok=True)

    token = os.environ.get('NGROK_AUTH_TOKEN') or os.environ.get('NGROK_AUTHTOKEN')
    if token:
        ngrok.set_auth_token(token)
    else:
        print('⚠️ NGROK auth token not set in environment; continuing without auth token')

    public_url = ngrok.connect(addr=5000)

    print("=" * 50)
    print(f"🔗 Public URL: {public_url}")
    print("📌 Prediction page: /predict-page")
    print("📌 Data collection page: /")
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) == 0:
        print(f"⚠️ Prediction disabled: {MODEL_PATH} is empty")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)