# Thai Digit Collector & Predictor

Web application for collecting handwritten Thai digit samples (๑๑, ๑๒, ๑๓, ๑๔, ๑๕) and predicting them using a deep learning model.

## Features

- **Data Collection**: Draw Thai digits (11-15) on canvas and save to dataset
- **Prediction**: AI model predicts drawn digits with confidence score
- **Public Access**: ngrok tunnel for remote access
- **Responsive**: Works on desktop and mobile

## Project Structure

```
Project_Collectdat/
├── app.py                  # Flask backend
├── requirements.txt        # Dependencies
├── thai_digit_model.h5     # Keras model for prediction
├── templates/
│   ├── index.html         # Data collection page
│   └── predict.html       # Prediction page
└── dataset/               # Collected images
    ├── 11/
    ├── 12/
    ├── 13/
    ├── 14/
    └── 15/
```

## Setup

```bash
pip install -r requirements.txt
```

### Requirements
- Flask
- pyngrok
- numpy
- pillow
- tensorflow
- keras

### ngrok Token (Optional)
```bash
# Windows PowerShell
$env:NGROK_AUTH_TOKEN = "your_token"
```

## Usage

```bash
python app.py
```

Open the displayed ngrok URL:

- **Data Collection**: `http://xxx.ngrok.io/`
- **Prediction**: `http://xxx.ngrok.io/predict-page`

## API

### POST `/save` - Save drawn image
```json
Request:  { "label": "11", "image": "base64_data" }
Response: { "status": "success", "filename": "11_123456.png" }
```

### POST `/predict` - Predict digit
```json
Request:  { "image": "base64_data" }
Response: { "predicted_class": 0, "thai_digit": 11, "confidence_score": 95.5 }
```

## Tech Stack

- Flask (Backend)
- HTML5 Canvas (Drawing)
- TensorFlow/Keras (ML Model)
- ngrok (Public URL)

## License

MIT