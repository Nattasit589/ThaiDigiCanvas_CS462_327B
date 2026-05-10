import base64
import os
import time
from io import BytesIO
from werkzeug.utils import secure_filename

import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from pyngrok import ngrok

app = Flask(__name__)

THAI_DIGIT_CLASSES = [11, 12, 13, 14, 15]
MODELS_DIR = 'models'
MODEL_PATH = 'thai_digit_model.h5'
_model = None
ALLOWED_EXTENSIONS = {'h5', 'keras'}
ACTIVE_MODEL = None  # Track which model is currently in use
ACTIVE_MODEL_FILE = os.path.join(MODELS_DIR, 'active_model.txt')


def get_available_models():
    """Get list of available models with their info"""
    models_list = []

    # Check models directory: expect each model in its own folder
    if os.path.exists(MODELS_DIR):
        for entry in sorted(os.listdir(MODELS_DIR)):
            entry_path = os.path.join(MODELS_DIR, entry)
            # skip non-dir files at the root (like active_model.txt)
            if os.path.isdir(entry_path):
                # look for any .h5/.keras inside
                for root, _, files in os.walk(entry_path):
                    for f in files:
                        if f.endswith(('.h5', '.keras')):
                            filepath = os.path.join(root, f)
                            try:
                                size = os.path.getsize(filepath)
                            except Exception:
                                size = 0
                            models_list.append({
                                'name': entry,
                                'file': f,
                                'path': filepath,
                                'source': 'uploaded',
                                'size_mb': round(size / (1024*1024), 2),
                                'location': entry_path
                            })
    # Check root fallback model
    if os.path.exists(MODEL_PATH):
        try:
            size = os.path.getsize(MODEL_PATH)
        except Exception:
            size = 0
        models_list.append({
            'name': MODEL_PATH,
            'file': os.path.basename(MODEL_PATH),
            'path': MODEL_PATH,
            'source': 'original',
            'size_mb': round(size / (1024*1024), 2),
            'location': 'root'
        })

    return models_list


def write_active_model(path):
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(ACTIVE_MODEL_FILE, 'w', encoding='utf-8') as fh:
            fh.write(path)
    except Exception:
        pass


def read_active_model():
    if os.path.exists(ACTIVE_MODEL_FILE):
        try:
            with open(ACTIVE_MODEL_FILE, 'r', encoding='utf-8') as fh:
                p = fh.read().strip()
                return p if p else None
        except Exception:
            return None
    return None


def load_prediction_model(force_reload=False, model_path=None):
    global _model, ACTIVE_MODEL
    if _model is not None and not force_reload and model_path is None:
        return _model

    # Determine model path priority:
    # 1) explicit model_path arg
    # 2) active model persisted in active_model.txt
    # 3) any model found in models/ (first)
    # 4) fallback to root MODEL_PATH
    model_to_load = None

    if model_path:
        model_to_load = model_path
    else:
        # check persisted active
        persisted = read_active_model()
        if persisted:
            model_to_load = persisted
        else:
            # pick first available in models/
            if os.path.exists(MODELS_DIR):
                for entry in sorted(os.listdir(MODELS_DIR)):
                    entry_path = os.path.join(MODELS_DIR, entry)
                    if os.path.isdir(entry_path):
                        for root, _, files in os.walk(entry_path):
                            for f in files:
                                if f.endswith(('.h5', '.keras')):
                                    model_to_load = os.path.join(root, f)
                                    break
                            if model_to_load:
                                break
                    if model_to_load:
                        break
    if model_to_load is None:
        model_to_load = MODEL_PATH

    if not os.path.exists(model_to_load):
        raise FileNotFoundError(f'Model file not found: {model_to_load}')

    if os.path.getsize(model_to_load) == 0:
        raise ValueError(f'Model file is empty: {model_to_load}')

    from keras.models import load_model

    _model = load_model(model_to_load, compile=False)
    ACTIVE_MODEL = model_to_load
    # persist active model
    write_active_model(model_to_load)
    return _model


def reload_model():
    """Force reload the model from disk"""
    global _model
    _model = None
    return load_prediction_model(force_reload=True)


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


@app.route('/admin')
def admin():
    """Admin page for model management"""
    return render_template('admin.html')


@app.route('/admin/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    try:
        models_list = get_available_models()
        return jsonify({
            'status': 'success',
            'models': models_list,
            'active_model': ACTIVE_MODEL or (os.path.join(MODELS_DIR, 'thai_digit_model.h5') if os.path.exists(os.path.join(MODELS_DIR, 'thai_digit_model.h5')) else MODEL_PATH)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/admin/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    data = request.get_json(silent=True)
    if not data or 'model_path' not in data:
        return jsonify({'error': 'No model path provided'}), 400
    
    model_path = data['model_path']
    
    try:
        # Verify model exists
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404
        
        # Load the model and persist selection
        load_prediction_model(force_reload=True, model_path=model_path)
        write_active_model(model_path)

        return jsonify({
            'status': 'success',
            'message': f'Switched to model: {model_path}',
            'active_model': model_path
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Model switch failed: {str(e)}'}), 500


@app.route('/admin/upload', methods=['POST'])
def admin_upload():
    """Handle model file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check file extension
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    
    if file_ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    try:
        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Create a subfolder per upload to preserve previous uploads
        timestamp = int(time.time() * 1000)
        name_only = os.path.splitext(filename)[0]
        folder_name = f"{name_only}_{timestamp}"
        folder_path = os.path.join(MODELS_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Save the uploaded model inside the folder as model.h5
        model_filename = 'model.h5'
        model_path = os.path.join(folder_path, model_filename)
        file.save(model_path)

        # Optionally save metadata
        try:
            meta = {'original_filename': filename, 'uploaded_at': timestamp}
            with open(os.path.join(folder_path, 'meta.json'), 'w', encoding='utf-8') as mf:
                import json
                json.dump(meta, mf)
        except Exception:
            pass

        # Automatically switch to the newly uploaded model and persist selection
        load_prediction_model(force_reload=True, model_path=model_path)
        write_active_model(model_path)

        return jsonify({
            'status': 'success',
            'message': f'Model uploaded successfully: {folder_name}/model.h5',
            'path': model_path,
            'active_model': model_path
        }), 200

    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


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
    os.makedirs(MODELS_DIR, exist_ok=True)

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
    print("📌 Admin page: /admin")
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) == 0:
        print(f"⚠️ Prediction disabled: {MODEL_PATH} is empty")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)