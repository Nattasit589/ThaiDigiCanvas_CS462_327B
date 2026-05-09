# Thai Digit Collector

A web application for collecting handwritten digit samples (Thai numerals 11-15) for machine learning dataset creation. Built with Flask and HTML5 Canvas.

## Features

- 🎨 **Canvas Drawing**: Draw digits with mouse or touch input
- 🏷️ **Label Selection**: Choose from digits 11, 12, 13, 14, 15
- 💾 **Automatic Storage**: Saves images to organized folders by label
- 🌐 **Public Access**: Uses ngrok for instant public URL tunneling
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
Project_Collectdat/
├── app.py              # Flask backend application
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Web interface with Canvas drawing
└── dataset/           # Output folder for collected images
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
- pillow: Image processing
- tensorflow: Deep learning framework for model loading

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
...
ngrok tunnel "http://xxx.ngrok.io" -> "http://localhost:5000"
```

Open the ngrok URL (or `http://localhost:5000` locally)

### 3. Collect Digits

1. Select a label (11-15) from the right panel
2. Draw a digit on the canvas with your mouse/touch
3. Click **Save** to store the image
4. Click **Clear** to draw again

## Image Storage

- Images are saved as PNG files with timestamps to prevent duplicates
- File naming: `{label}_{timestamp}.png`
- Organized in folders: `dataset/{label}/`
- Example: `dataset/11/11_1234567890123.png`

## API

### POST `/save`

Send drawn digit image to be saved.

**Request:**
```json
{
  "label": "11",
  "image": "base64_encoded_png_data"
}
```

**Response (Success):**
```json
{
  "status": "success",
  "filename": "11_1234567890123.png"
}
```

**Response (Error):**
```json
{
  "error": "Invalid data"
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

## Development Notes

- Canvas draws at 280x280 pixels (MNIST compatible)
- White strokes on black background
- Uses 15px line width for smooth writing
- Touch and mouse input both supported

## License

MIT

## Author

Created for Thai digit dataset collection project
