# Thai Digit Collector & Predictor

A Flask web application with two modes:
1. **Data Collection** - Collect handwritten digit samples (Thai numerals 11-15) for ML dataset creation
2. **AI Prediction** - Real-time handwritten digit recognition using a pre-trained CNN model

Built with Flask, HTML5 Canvas, NumPy, and Keras.

## Features

- 🎨 **Canvas Drawing**: Draw digits with mouse or touch input
- 🏷️ **Label Selection**: Choose from digits 11, 12, 13, 14, 15
- 💾 **Automatic Storage**: Saves images to organized folders by label
- 🔮 **AI Prediction**: Real-time digit recognition with confidence scores
- 🌐 **Public Access**: Uses ngrok for instant public URL tunneling
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
Project_Collectdat/
├── app.py                    # Flask backend with data collection & prediction routes
├── requirements.txt          # Python dependencies
├── thai_digit_model.h5       # Pre-trained Keras model (required for /predict)
├── templates/
│   ├── index.html           # Data collection interface (Canvas drawing + saving)
│   └── predict.html         # AI prediction interface (Draw → Predict)
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
- keras: Deep learning framework (for model loading and prediction)

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

1. Select a label (11-15) from the right panel
2. Draw a digit on the canvas with your mouse/touch
3. Click **Save** to store the image
4. Click **Clear** to draw again

### 4. AI Prediction

**Page:** `http://localhost:5000/predict-page`

1. Draw a digit on the canvas
2. Click **🔮 Predict** button
3. View the predicted digit class and confidence score

**Note:** Prediction requires a valid `thai_digit_model.h5` file (not empty). Currently showing placeholder until model is loaded.

## Image Storage

- Images are saved as PNG files with timestamps to prevent duplicates
- File naming: `{label}_{timestamp}.png`
- Organized in folders: `dataset/{label}/`
- Example: `dataset/11/11_1234567890123.png`

## Routes & API

### GET `/`

Serves the data collection interface (`index.html`).

### GET `/predict-page`

Serves the prediction interface (`predict.html`).

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
3. **Output shape** should be 5 classes (indices 0-4 mapped to digits 11-15)
4. The model is lazy-loaded on first prediction request

If the model file is missing or empty, the `/predict` endpoint returns HTTP 503 with a clear message.

## Development Notes

- Canvas draws at 280x280 pixels (MNIST compatible)
- White strokes on black background
- Image preprocessing: grayscale → 28×28 resize → normalize [0, 1]
- Touch and mouse input both supported
- Uses Tailwind CSS for responsive UI
- Model loading is lazy (loaded only when first prediction is made)

## License

MIT

## Author

Created for Thai digit dataset collection project
