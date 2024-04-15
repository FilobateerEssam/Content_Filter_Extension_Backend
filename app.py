from flask import Flask, render_template, request , jsonify
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
from PIL import Image
import sys
import requests
import io
from io import BytesIO
import json
import pandas as pd
from string import digits
import re
import unicodedata
import keras_nlp
import numpy as np


# Load the model with custom objects
mobileNet_image_model = tf.saved_model.load("models/MobileNetV3")

efficientNet_image_model = tf.saved_model.load("models/EfficientNet")

distilBert_text_model=tf.saved_model.load('models/distilbert_classifier')

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "static/Uploads/"

@app.route("/")
def home():
    return render_template("helloWorld.html")

@app.route("/upload-image", methods=["","POST"])
def upload_image():
    if request.method == "POST":
        if "image" in request.files:
            image = request.files["image"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            filename = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
            predict_image = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
            predict_image = tf.keras.preprocessing.image.img_to_array(predict_image)
            predict_image = tf.expand_dims(predict_image, axis=0)
            prediction=mobileNet_image_model.predict(predict_image)
            if prediction[0][0]<prediction[0][1]:
                prediction="Violence"
            else:
                prediction="Non-Violence"

            return render_template("upload_image.html", uploaded_image=filename, model_prediction=prediction )
    return render_template("upload_image.html")

@app.route('/process-image', methods=["POST"])
def process_image():
 if request.method == "POST":
    if "image" in request.files:
        image = request.files["image"]
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
        filename = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
        predict_image = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
        predict_image = tf.keras.preprocessing.image.img_to_array(predict_image)
        predict_image = tf.expand_dims(predict_image, axis=0)
        prediction=mobileNet_image_model.predict(predict_image)
        if prediction[0][0]<prediction[0][1]:
            prediction="Violence"
        else:
            prediction="Non-Violence"

        return jsonify({'msg': 'success', 'prediction': [prediction]})
    return jsonify({'Error': 'BackendError'})

@app.route('/test', methods=["POST"])
def test():
    if request.method == "POST":
         if "image" in request.files:
             return jsonify({'msg': 'Bravo'})
         return jsonify({'msg': 'Bad'})


@app.route('/upload-urls',methods=["POST"])
def getImagesList():
    if request.method =="POST":
        if "images" in request.form:
            images=request.form['images']
            # img=json.loads(images)
            images=images.split(',')
            # return jsonify({'msg': 'success', 'images-recived':images})
            predictions={}
            multi_class_predictions={}
            for image in images:
                if image=="":
                    continue
                response = requests.get(image)
                img = Image.open(io.BytesIO(response.content)).resize((224,224)).convert('RGB')
                # img.save(os.path.join(app.config["IMAGE_UPLOADS"], img.filename))
                # filename = os.path.join(app.config["IMAGE_UPLOADS"], img.filename)
                # predict_image = tf.keras.preprocessing.image.load_img(io.BytesIO(response.content), target_size=(224, 224))
                predict_image = tf.keras.preprocessing.image.img_to_array(img)
                predict_image = tf.expand_dims(predict_image, axis=0)

                prediction=mobileNet_image_model.signatures["serving_default"](predict_image)
                eff_input=tf.keras.applications.efficientnet_v2.preprocess_input(predict_image)
               # input_tensor = tf.convert_to_tensor(eff_input, dtype=tf.float32)
                predic = efficientNet_image_model.signatures["serving_default"](eff_input)
                # Extracting the numpy array from the tensor
                non_violence,violence=prediction['dense'].numpy()[0]
                prediction_array = predic['output_0'].numpy()

                # Assigning each prediction to a separate variable
                accident, damaged_buildings, fire, normal = prediction_array[0]
                predictions_dict={"fire":fire,"accident":accident,'normal':normal,"damaged_buildings":damaged_buildings}
                print(prediction)
                if non_violence<violence:
                    prediction="Violence"
                else:
                    prediction="Non-Violence"
                predictions[image]=prediction
                print("prediction DEC:")
                print(predictions_dict)
                print(max(predictions_dict,key=predictions_dict.get))
                multi_class_predictions[image]=max(predictions_dict,key=predictions_dict.get)
            return jsonify({'msg': 'success', 'prediction': predictions,'mulit-class-prediction':multi_class_predictions})
        return jsonify({"BackendError":"Images not sent"})
    return jsonify({"BackendError": "Error in request"})

@app.route('/upload-text',methods=['POST'])
def getStringsList():
    print("GetStringListCalled")
    if request.method =="POST":
        if 'textData' in request.files:
            print("Call .files instead")
            return jsonify( {"Text Res":"call .files"})
        if 'textData' in request.form:
            textData=request.form['textData']
            textList=textData.split(",")
            # for text in textList:
            #     predictions=distilBert_text_model.signatures["serving_default"](text)
            #     print(predictions)
            return jsonify( {"Text Res":"predictions"})
        return jsonify({"Error":"No data recived"})
    return jsonify({"Error ":"Wrong request"})

if __name__ == "__main__":
    app.run(debug=True)
