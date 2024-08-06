# pip install flask onnxruntime opencv-python-headless pillow numpy tqdm
# run this in terminal before running the script




from flask import Flask, request, render_template, send_file, redirect, url_for
import onnxruntime as ort
import cv2
import numpy as np
import os
from PIL import Image

app = Flask(__name__, static_url_path='/outputs', static_folder='outputs')


pic_form = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG','.webp','.WEBP']
device_name = ort.get_device()

if device_name == 'CPU':
    providers = ['CPUExecutionProvider']
elif device_name == 'GPU':
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# model = 'Paprika_54'
# session = ort.InferenceSession(f'./{model}.onnx', providers=providers)

def process_image(img, x32=True):
    h, w = img.shape[:2]
    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return img

def load_test_data(image_path):
    img0 = cv2.imread(image_path).astype(np.float32)
    img = process_image(img0)
    img = np.expand_dims(img, axis=0)
    return img, img0.shape[:2]

def Convert(img, scale,model):
    session = ort.InferenceSession(f'./{model}.onnx', providers=providers)
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name
    fake_img = session.run(None, {x: img})[0]
    images = (np.squeeze(fake_img) + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    output_image = cv2.resize(images, (scale[1], scale[0]))
    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    model=request.form['model']
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        img, scale = load_test_data(filepath)
        output_img = Convert(img, scale, model)
        output_path = os.path.join('outputs', filename)
        cv2.imwrite(output_path, output_img)

        return render_template('index.html',filename=filename)

def allowed_file(filename):
    return any(filename.endswith(ext) for ext in pic_form)

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
