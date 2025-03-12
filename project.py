from flask import Flask,render_template,request,url_for
import numpy as np
import cv2
import tensorflow as tf
import os

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        model = tf.keras.models.load_model("mobile_net.h5")
        file = request.files["image"]
        if file.filename == "":
            return "No selected file"
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
        pred_img = cv2.imread(image_path)
        pred_image = cv2.resize(pred_img,(224,224))
        pred_image = cv2.cvtColor(pred_image,cv2.COLOR_BGR2RGB)
        pred_image = np.array(pred_image)
        pred_image = np.expand_dims(pred_image,axis=0)
        model.predict(pred_image)
        label = np.argmax(model.predict(pred_image))

        label_text = "AI" if label == 0 else "Real"

        image_url = url_for('static', filename=f"uploads/{file.filename}")
        return render_template('result.html', image_path=image_url, prediction=label_text)
    return render_template('linear_regression.html')


if __name__=="__main__":
    app.run(debug=True)