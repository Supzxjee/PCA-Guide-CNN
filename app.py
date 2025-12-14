import os
import cv2
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('model_CK+.h5')

with open('pca_CK+.pkl', 'rb') as f:
    pca = pickle.load(f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EMOTION_DICT = {
    0: "Lỗi (File Rác)",          #
    1: "Angry (Tức giận)",       
    2: "Contempt (Khinh thường)", 
    3: "Disgust (Ghê tởm)",       
    4: "Fear (Sợ hãi)",      
    5: "Happy (Vui vẻ)",          
    6: "Sadness (Buồn bã)",    
    7: "Surprise (Ngạc nhiên)"   
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files: return jsonify({'error': 'Chưa upload file'}), 400
        file = request.files['file']
        
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img_original = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE) 
        
        if img_original is None: return jsonify({'error': 'File lỗi'}), 400

        height, width = img_original.shape
        
        if height < 60 or width < 60:
            img_roi = img_original
            faces = [] 
        else:
            faces = face_cascade.detectMultiScale(img_original, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                img_roi = img_original[y:y+h, x:x+w]
            else:
                img_roi = img_original
        
        faces = face_cascade.detectMultiScale(img_original, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        img_roi = None
        
        if len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            img_roi = img_original[y:y+h, x:x+w]
        else:
            img_roi = img_original

        img_resized = cv2.resize(img_roi, (48, 48))
        
        img_normalized = img_resized.astype('float32') / 255.0
        
        img_flat = img_normalized.reshape(1, -1)
        img_pca = pca.inverse_transform(pca.transform(img_flat)).reshape(1, 48, 48, 1)

        prediction = model.predict(img_pca)
        max_index = np.argmax(prediction)

        predicted_emotion = EMOTION_DICT[max_index]
        confidence = float(np.max(prediction)) * 100

        return jsonify({
            'status': 'success',
            'dominant_emotion': predicted_emotion,
            'confidence': f"{confidence:.2f}%",
            'face_detected': len(faces) > 0
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)