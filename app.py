import flask
import pickle
from flask import Flask, render_template, url_for, redirect, request, flash
from werkzeug.utils import secure_filename
import tensorflow
import keras 
import os
import pickle
import pandas as pd
import numpy as np

UPLOAD_FOLDER = 'static/images/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
model = keras.models.load_model('image_seg_model_10class.h5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

label_img = {0:'Airplane',1:'Automobile',2:'Bird',3:'Cat',4:'Deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predicted_img(loc):
    img = np.array(tensorflow.keras.preprocessing.image.load_img(loc, target_size=(32, 32, 3)))
    img = np.expand_dims(img, axis=0)
    predicted = model.predict(img)
    return label_img[int(np.argmax(predicted, axis=1))]

@app.route('/', methods = ["GET", "POST"])
def home():
    if request.method == "POST":
        image = request.files

        if 'file' not in request.files:
            return render_template("index.html", filename='No file part')

        file = request.files['file']

        if file.filename == '':
            re_type = 'No selected file'
            return render_template("index.html", model_output=re_type)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        model_predicted = predicted_img("static/images/"+filename)
        return render_template("index.html", filename=filename, model_output='Its an ' + model_predicted)

    return render_template("index.html", model_output='')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename = "static/images/"+filename))

if __name__ == "__main__":
    app.run(debug=True)