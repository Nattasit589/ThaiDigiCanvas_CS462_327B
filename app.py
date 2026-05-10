import base64
import json
import os
import shutil
import time
from io import BytesIO
from werkzeug.utils import secure_filename

import h5py
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


def _decode_h5_attr(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def _load_layer_weights(model_weights_group, layer_name):
    layer_group = model_weights_group[layer_name]

    def _find_weight_group(group):
        for key in group.keys():
            candidate = group[key]
            if isinstance(candidate, h5py.Group):
                if 'kernel' in candidate and 'bias' in candidate:
                    return candidate
                nested = _find_weight_group(candidate)
                if nested is not None:
                    return nested
        return None

    weight_group = _find_weight_group(layer_group)
    if weight_group is None:
        raise ValueError(f'Could not find weights for layer: {layer_name}')

    return {
        'kernel': np.array(weight_group['kernel'], dtype=np.float32),
        'bias': np.array(weight_group['bias'], dtype=np.float32),
    }


def _load_batchnorm_weights(model_weights_group, layer_name):
    """Find and load BatchNormalization weights for a given layer name."""
    layer_group = model_weights_group[layer_name]

    def _find_bn_group(group):
        for key in group.keys():
            candidate = group[key]
            if isinstance(candidate, h5py.Group):
                keys = set(candidate.keys())
                if 'gamma' in keys or 'beta' in keys or 'moving_mean' in keys or 'moving_variance' in keys:
                    return candidate
                nested = _find_bn_group(candidate)
                if nested is not None:
                    return nested
        return None

    weight_group = _find_bn_group(layer_group)
    if weight_group is None:
        raise ValueError(f'Could not find BatchNormalization weights for layer: {layer_name}')

    def _get(name):
        return np.array(weight_group[name], dtype=np.float32) if name in weight_group else None

    return {
        'gamma': _get('gamma'),
        'beta': _get('beta'),
        'moving_mean': _get('moving_mean'),
        'moving_variance': _get('moving_variance'),
    }


def _apply_activation(values, activation):
    if activation in (None, '', 'linear'):
        return values
    if activation == 'relu':
        return np.maximum(values, 0.0)
    if activation == 'softmax':
        shifted = values - np.max(values, axis=-1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    raise ValueError(f'Unsupported activation: {activation}')


def _conv2d_valid(batch_input, kernel, bias, strides):
    batch_size, input_height, input_width, input_channels = batch_input.shape
    kernel_height, kernel_width, kernel_channels, filters = kernel.shape
    stride_y, stride_x = strides

    if input_channels != kernel_channels:
        raise ValueError('Input channels do not match convolution kernel channels')

    output_height = (input_height - kernel_height) // stride_y + 1
    output_width = (input_width - kernel_width) // stride_x + 1
    output = np.empty((batch_size, output_height, output_width, filters), dtype=np.float32)
    kernel_flat = kernel.reshape(-1, filters)

    for batch_index in range(batch_size):
        for row_index in range(output_height):
            row_start = row_index * stride_y
            row_end = row_start + kernel_height
            for col_index in range(output_width):
                col_start = col_index * stride_x
                col_end = col_start + kernel_width
                patch = batch_input[batch_index, row_start:row_end, col_start:col_end, :].reshape(-1)
                output[batch_index, row_index, col_index, :] = patch @ kernel_flat + bias

    return output


def _max_pool2d_valid(batch_input, pool_size, strides):
    batch_size, input_height, input_width, input_channels = batch_input.shape
    pool_height, pool_width = pool_size
    stride_y, stride_x = strides

    output_height = (input_height - pool_height) // stride_y + 1
    output_width = (input_width - pool_width) // stride_x + 1
    output = np.empty((batch_size, output_height, output_width, input_channels), dtype=np.float32)

    for batch_index in range(batch_size):
        for row_index in range(output_height):
            row_start = row_index * stride_y
            row_end = row_start + pool_height
            for col_index in range(output_width):
                col_start = col_index * stride_x
                col_end = col_start + pool_width
                window = batch_input[batch_index, row_start:row_end, col_start:col_end, :]
                output[batch_index, row_index, col_index, :] = np.max(window, axis=(0, 1))

    return output


class NumpySequentialModel:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input_tensor, verbose=0):
        values = np.asarray(input_tensor, dtype=np.float32)

        for layer in self.layers:
            layer_type = layer['type']
            if layer_type == 'activation':
                values = _apply_activation(values, layer.get('activation'))
                continue
            if layer_type == 'conv2d':
                values = _conv2d_valid(values, layer['kernel'], layer['bias'], layer['strides'])
                values = _apply_activation(values, layer['activation'])
            elif layer_type == 'max_pooling2d':
                values = _max_pool2d_valid(values, layer['pool_size'], layer['strides'])
            elif layer_type == 'flatten':
                values = values.reshape(values.shape[0], -1)
            elif layer_type == 'batch_normalization':
                # Apply batch normalization using stored moving statistics
                gamma = layer.get('gamma')
                beta = layer.get('beta')
                mean = layer.get('moving_mean')
                var = layer.get('moving_variance')
                eps = float(layer.get('epsilon', 1e-3))

                if mean is None or var is None:
                    # nothing to do
                    continue

                if values.ndim == 4:
                    # broadcast (1,1,1,channels)
                    mean_r = mean.reshape((1, 1, 1, -1))
                    var_r = var.reshape((1, 1, 1, -1))
                    gamma_r = (gamma.reshape((1, 1, 1, -1)) if gamma is not None else 1.0)
                    beta_r = (beta.reshape((1, 1, 1, -1)) if beta is not None else 0.0)
                    values = (values - mean_r) / np.sqrt(var_r + eps) * gamma_r + beta_r
                elif values.ndim == 2:
                    mean_r = mean.reshape((1, -1))
                    var_r = var.reshape((1, -1))
                    gamma_r = (gamma.reshape((1, -1)) if gamma is not None else 1.0)
                    beta_r = (beta.reshape((1, -1)) if beta is not None else 0.0)
                    values = (values - mean_r) / np.sqrt(var_r + eps) * gamma_r + beta_r
                else:
                    # fallback: apply elementwise if shapes allow
                    try:
                        values = (values - mean) / np.sqrt(var + eps) * (gamma if gamma is not None else 1.0) + (beta if beta is not None else 0.0)
                    except Exception:
                        pass
            elif layer_type == 'global_avg_pooling2d':
                # values shape expected: (batch, height, width, channels)
                if values.ndim == 4:
                    values = np.mean(values, axis=(1, 2))
                elif values.ndim == 3:
                    # (batch, height, channels) -> mean over spatial dim
                    values = np.mean(values, axis=1)
                else:
                    # leave as-is if unexpected shape
                    try:
                        values = np.mean(values, axis=tuple(range(1, values.ndim)))
                    except Exception:
                        pass
            elif layer_type == 'dense':
                values = values @ layer['kernel'] + layer['bias']
                values = _apply_activation(values, layer['activation'])
            else:
                raise ValueError(f'Unsupported layer type: {layer_type}')

        return values


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

    with h5py.File(model_to_load, 'r') as h5_file:
        model_config = json.loads(_decode_h5_attr(h5_file.attrs['model_config']))
        if model_config.get('class_name') != 'Sequential':
            raise ValueError('Only Sequential H5 models are supported by this prediction backend')

        layer_specs = model_config['config']['layers']
        model_weights = h5_file['model_weights']
        layers = []

        for layer_spec in layer_specs:
            layer_type = layer_spec['class_name']
            layer_name = layer_spec['config'].get('name')

            if layer_type in ('InputLayer', 'Dropout'):
                continue
            if layer_type == 'Conv2D':
                weights = _load_layer_weights(model_weights, layer_name)
                layers.append({
                    'type': 'conv2d',
                    'kernel': weights['kernel'],
                    'bias': weights['bias'],
                    'strides': tuple(layer_spec['config'].get('strides', [1, 1])),
                    'activation': layer_spec['config'].get('activation', 'linear'),
                })
            elif layer_type == 'MaxPooling2D':
                layers.append({
                    'type': 'max_pooling2d',
                    'pool_size': tuple(layer_spec['config'].get('pool_size', [2, 2])),
                    'strides': tuple(layer_spec['config'].get('strides', [2, 2])),
                })
            elif layer_type == 'Flatten':
                layers.append({'type': 'flatten'})
            elif layer_type == 'GlobalAveragePooling2D':
                layers.append({'type': 'global_avg_pooling2d'})
            elif layer_type == 'Dense':
                weights = _load_layer_weights(model_weights, layer_name)
                layers.append({
                    'type': 'dense',
                    'kernel': weights['kernel'],
                    'bias': weights['bias'],
                    'activation': layer_spec['config'].get('activation', 'linear'),
                })
            elif layer_type == 'Activation':
                # standalone Activation layer (e.g., Activation('relu'))
                layers.append({
                    'type': 'activation',
                    'activation': layer_spec['config'].get('activation', 'linear')
                })
            elif layer_type == 'BatchNormalization':
                # load batchnorm params (gamma, beta, moving_mean, moving_variance)
                bn = _load_batchnorm_weights(model_weights, layer_name)
                epsilon = layer_spec['config'].get('epsilon', 1e-3)
                layers.append({
                    'type': 'batch_normalization',
                    'gamma': bn.get('gamma'),
                    'beta': bn.get('beta'),
                    'moving_mean': bn.get('moving_mean'),
                    'moving_variance': bn.get('moving_variance'),
                    'epsilon': float(epsilon),
                })
            else:
                raise ValueError(f'Unsupported model layer: {layer_type}')

    _model = NumpySequentialModel(layers)
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


@app.route('/admin/delete', methods=['POST'])
def delete_model():
    """Delete a model folder and reset active_model if necessary"""
    global _model, ACTIVE_MODEL
    data = request.get_json(silent=True)
    if not data or 'model_path' not in data:
        return jsonify({'error': 'No model path provided'}), 400
    
    model_path = data['model_path']
    
    try:
        # Security check: only allow deletion of models inside MODELS_DIR
        # (not the root thai_digit_model.h5)
        model_abs_path = os.path.abspath(model_path)
        models_abs_dir = os.path.abspath(MODELS_DIR)
        
        if not model_abs_path.startswith(models_abs_dir):
            return jsonify({'error': 'Cannot delete models outside the models directory'}), 403
        
        # Check if the model/folder exists
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 404
        
        # If this is the active model, reset to default
        persisted_active = read_active_model()
        if persisted_active and os.path.abspath(persisted_active) == model_abs_path:
            write_active_model(MODEL_PATH)  # Reset to default
            _model = None  # Unload the model
            ACTIVE_MODEL = None
        
        # Delete the folder or file
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        else:
            os.remove(model_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Model deleted successfully: {model_path}',
            'reset_to_default': persisted_active and os.path.abspath(persisted_active) == model_abs_path
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Delete failed: {str(e)}'}), 500


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