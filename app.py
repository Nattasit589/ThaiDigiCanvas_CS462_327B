import base64
import os
import time
from flask import Flask, request, jsonify, render_template
from pyngrok import ngrok

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save():
    data = request.get_json()
    if not data or 'label' not in data or 'image' not in data:
        return jsonify({'error': 'Invalid data'}), 400

    label = str(data['label'])
    img_b64 = data['image']

    # สร้างโฟลเดอร์ dataset/11, dataset/12 ฯลฯ
    folder = os.path.join('dataset', label)
    os.makedirs(folder, exist_ok=True)

    # ชื่อไฟล์จาก timestamp (กันซ้ำ)
    timestamp = int(time.time() * 1000)
    filename = f"{label}_{timestamp}.png"
    filepath = os.path.join(folder, filename)

    with open(filepath, 'wb') as f:
        f.write(base64.b64decode(img_b64))

    return jsonify({'status': 'success', 'filename': filename})

if __name__ == '__main__':
    os.makedirs('dataset', exist_ok=True)

    # อ่าน ngrok authtoken จาก environment variable (ถ้ามี)
    # สนับสนุนทั้ง NGROK_AUTH_TOKEN และ NGROK_AUTHTOKEN
    token = os.environ.get('NGROK_AUTH_TOKEN') or os.environ.get('NGROK_AUTHTOKEN')
    if token:
        ngrok.set_auth_token(token)
    else:
        print('⚠️ NGROK auth token not set in environment; continuing without auth token')

    # เปิด ngrok tunnel (no basic auth)
    public_url = ngrok.connect(addr=5000)

    print("=" * 50)
    print(f"🔒 Secure URL: {public_url}")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)