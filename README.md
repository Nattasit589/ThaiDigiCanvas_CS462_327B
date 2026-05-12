# Thai Digit Collector & Predictor

A Flask web application with two modes:
1. **Data Collection** - Collect handwritten digit samples (Thai numerals ๑๑–๑๕) for ML dataset creation
2. **AI Prediction** - Real-time handwritten digit recognition using a pre-trained CNN model

Built with Flask, HTML5 Canvas, NumPy, and Keras.

## Features

- 🎨 **Canvas Drawing**: Draw digits with mouse or touch input
- 🏷️ **Thai Numeral Labels**: UI displays ๑๑, ๑๒, ๑๓, ๑๔, ๑๕ (Thai numerals)
- 💾 **Automatic Storage**: Saves images to organized folders by label with timestamps
- 🔮 **AI Prediction**: Real-time digit recognition with confidence scores
- 📊 **Admin Panel**: Upload and switch between different trained models
- 🌐 **Public Access**: Uses ngrok for instant public URL tunneling
- 📱 **Responsive Design**: Works on desktop and mobile devices with touch support

## Project Structure

```
Project_Collectdat/
├── app.py                    # Flask backend with all routes (collection, prediction, admin)
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── thai_digit_model.h5       # Pre-trained Keras model (required for /predict)
├── models/                   # Uploaded model storage
│   └── active_model.txt      # Tracks currently active model path
├── templates/
│   ├── index.html           # Data collection interface (Canvas drawing + saving)
│   ├── predict.html         # AI prediction interface (Draw → Predict) with Thai numerals
│   └── admin.html           # Admin panel for model management
└── dataset/                 # Output folder for collected images
    ├── 11/
    ├── 12/
    ├── 13/
    ├── 14/
    └── 15/
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Flask: Web framework
- flask-httpauth: HTTP authentication
- pyngrok: ngrok tunneling client
- werkzeug: WSGI utilities
- numpy: Numerical computing
- pillow: Image processing (PIL)
- h5py: Reads the saved Keras `.h5` model weights for prediction

### 2. Get ngrok Auth Token (Optional but Recommended)

1. Sign up at [ngrok.com](https://ngrok.com)
2. Copy your auth token from the dashboard
3. Set environment variable:
   ```bash
   # Windows (PowerShell)
   $env:NGROK_AUTH_TOKEN="your_token_here"
   
   # Windows (CMD)
   set NGROK_AUTH_TOKEN=your_token_here
   ```

## Usage

### 1. Start the Application

```bash
python app.py
```

### 2. Open in Browser

The app will display:
```
==================================================
🔗 Public URL: https://xxx.ngrok-free.dev
📌 Prediction page: /predict-page
📌 Data collection page: /
⚠️ Prediction disabled: thai_digit_model.h5 is empty
==================================================
```

Open the ngrok URL (or `http://localhost:5000` locally)

### 3. Data Collection

**Page:** `http://localhost:5000/`

1. Select a label (๑๑–๑๕) from the right panel
2. Draw a digit on the canvas with your mouse/touch
3. Click **Save** to store the image
4. Click **Clear** to draw again

### 4. AI Prediction

**Page:** `http://localhost:5000/predict-page`

1. Draw a digit on the canvas
2. Click **🔮 Predict** button
3. View the predicted digit (displayed in Thai numerals) and confidence score

**Note:** Prediction requires a valid `thai_digit_model.h5` file (not empty). The main digit result displays in Thai numerals, while class and confidence remain in Arabic numerals.

### 5. Admin Panel

**Page:** `http://localhost:5000/admin`

1. **Upload Model**: Select a `.h5` or `.keras` file to upload
2. **View Available Models**: See list of uploaded models with file sizes
3. **Switch Models**: Click "Switch" to activate a different model for predictions
4. **Delete Models**: Remove models from the `models/` directory (keeps root model safe)

**Note:** Newly uploaded models are automatically activated and their path is persisted in `active_model.txt`.

## Image Storage

- Images are saved as PNG files with timestamps to prevent duplicates
- File naming: `{label}_{timestamp}.png`
- Organized in folders: `dataset/{label}/`
- Example: `dataset/11/11_1234567890123.png`

## Routes & API

### GET `/`

Serves the data collection interface (`index.html`).

### GET `/predict-page`

Serves the prediction interface (`predict.html`) with Thai numeral display.

### GET `/admin`

Serves the admin panel interface (`admin.html`) for model management.

### GET `/admin/models`

Fetch list of available models and current active model.

**Response (Success - 200):**
```json
{
  "status": "success",
  "models": [
    {
      "name": "model_1234567890",
      "file": "model.h5",
      "path": "models/model_1234567890/model.h5",
      "source": "uploaded",
      "size_mb": 45.32,
      "location": "models/model_1234567890"
    }
  ],
  "active_model": "models/model_1234567890/model.h5"
}
```

### POST `/admin/upload`

Upload a new model file and automatically switch to it.

**Request:** multipart/form-data with `file` field (`.h5` or `.keras`)

**Response (Success - 200):**
```json
{
  "status": "success",
  "message": "Model uploaded successfully: model_name_1234567890/model.h5",
  "path": "models/model_name_1234567890/model.h5",
  "active_model": "models/model_name_1234567890/model.h5"
}
```

### POST `/admin/switch`

Switch the active model for predictions.

**Request:**
```json
{
  "model_path": "models/model_name_1234567890/model.h5"
}
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "message": "Switched to model: models/model_name_1234567890/model.h5",
  "active_model": "models/model_name_1234567890/model.h5"
}
```

### POST `/admin/delete`

Delete a model folder (only from `models/` directory, root model is protected).

**Request:**
```json
{
  "model_path": "models/model_name_1234567890"
}
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "message": "Model deleted successfully: models/model_name_1234567890",
  "reset_to_default": false
}
```

### POST `/save`

Save a drawn digit image to the dataset.

**Request:**
```json
{
  "label": "11",
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "filename": "11_1234567890123.png"
}
```

**Response (Error - 400):**
```json
{
  "error": "Invalid data"
}
```

### POST `/predict`

Predict a Thai digit from a canvas image.

**Request:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

**Response (Success - 200):**
```json
{
  "predicted_class": 0,
  "thai_digit": 11,
  "confidence_score": 95.42
}
```

Note: The UI displays the `thai_digit` (11-15) in Thai numerals (๑๑–๑๕) while keeping class and confidence score in Arabic numerals
```

**Response (Model unavailable - 503):**
```json
{
  "error": "Prediction unavailable: Model file is empty: thai_digit_model.h5"
}
```

**Response (Error - 500):**
```json
{
  "error": "Prediction failed: <error message>"
}
```

## Security Notes

⚠️ **Important**: When exposing via ngrok, be aware that:
- The application has no authentication enabled
- Anyone with the URL can save files
- Consider adding authentication for production use
- The app runs locally and tunnels through ngrok

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ngrok connection fails | Check internet connection or set auth token |
| Canvas doesn't respond | Try refreshing the page or checking browser console |
| Images not saving | Verify `dataset/` folder has write permissions |
| No ngrok tunnel | Ensure pyngrok is installed: `pip install pyngrok` |

## Model Setup

### Loading the Model for Predictions

The prediction feature requires a valid Keras model file (`thai_digit_model.h5`):

1. **Place your trained model** at the project root: `thai_digit_model.h5`
2. **Ensure it accepts** 28×28 grayscale input (MNIST format)
3. **Output shape** should be 5 classes (indices 0-4 mapped to digits ๑๑–๑๕)
4. The model is lazy-loaded on first prediction request
5. Prediction runs with a NumPy/H5 backend, so TensorFlow is not required at runtime

If the model file is missing or empty, the `/predict` endpoint returns HTTP 503 with a clear message.

Supported layers and runtime notes:
- The prediction backend now supports additional Keras layers commonly used in Sequential CNNs: `BatchNormalization`, standalone `Activation` layers (e.g., `Activation('relu')`), and `GlobalAveragePooling2D`.
- Batch normalization weights (`gamma`, `beta`, `moving_mean`, `moving_variance`) are read from the `.h5` and applied using stored moving statistics (inference mode).
- If your trained model uses custom layers not listed above, the backend will still raise an `Unsupported model layer` error — convert or remove those layers before exporting.

## Development Notes

- Canvas draws at 280x280 pixels (MNIST compatible)
- White strokes on black background
- Image preprocessing: grayscale → 28×28 resize → normalize [0, 1]
- Touch and mouse input both supported
- Uses Tailwind CSS for responsive UI
- Model loading is lazy (loaded only when first prediction is made)
- Thai numeral conversion applied only in UI (backend stays Arabic for compatibility)
- Model paths persisted in `models/active_model.txt` across restarts

## UI Features

### Thai Numeral Display (v1.1)
- **Data Collection page**: Shows labels as ๑๑–๑๕ instead of 11–15
- **Prediction page**: Displays predicted digit in Thai numerals (๑๑–๑๕)
- **Confidence & Class**: Kept in Arabic numerals for clarity
- Conversion function: `toThaiDigits()` in JavaScript (`๐๑๒๓๔๕๖๗๘๙` mapping)

## Changelog

### v1.1 (Latest)
- ✅ Added Thai numeral display throughout UI
- ✅ Complete Admin panel for model management
- ✅ Updated README with full documentation
- ✅ Team member credits added

### v1.0
- ✅ Initial release with data collection and prediction
- ✅ Canvas drawing with mouse and touch support
- ✅ Image storage with timestamps
- ✅ NumPy-based model inference (no TensorFlow required at runtime)

## Team Members

| Name | Student ID |
|------|------------|
| ณัฐสิทธิ์ สงวนธนากร | 1660701390 |
| ธนธัส คันทรินทร์ | 1670700663 |
| พัชรพล ชุลีประเสริฐ | 1660707462 |
| ภาคิน เชาว์โคกสูง | 1660700046 |
| นายฐานันดร ศรีสวัสดิ์ | 1660707041 |
| นายภควัต สุขสมโสตร | 1660704279 |

## License

MIT

## Project

Created for Thai digit dataset collection and recognition project - CS462_327B
