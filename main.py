# import the necessary packages
#from keras.models import load_model
from tensorflow.keras.models import load_model
import numpy as np
import flask
import cv2
import time
import datetime as DT
import pandas as pd
from statistics import mean
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

model=load_model('my_model_roi_nasnet_trained.h5',compile = False)

def roi_lenet(image):
    #image_path = "savedimg/"+str(file_name)
    #image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 240))
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    startX, startY, endX, endY = pred[0]
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    box = [startX,startY,endX,endY]
    return box

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}
    model = 0
    # ensure an data was properly uploaded to our endpoint
    if flask.request.method == "POST":

        file = flask.request.files.get('file')
        file.save('data/'+file.filename)
        image = cv2.imread('data/' + file.filename)
        print('data/' + file.filename)
        device_id = file.filename.replace('.jpg','')
        start = time.time()
        box = roi_lenet(image)
        #box =[0,0,20,20]
        end = time.time()
        print("[INFO] text detection took {:.6f} seconds".format(end - start))

        print(box)

        startX, startY, endX, endY = box

        height = endY - startY
        width = endX - startX
        data['device_id'] = device_id
        data['startX'] = startX
        data['endX'] = endX
        data['startY'] = startY
        data['endY'] = endY
        data['height'] = height
        data['width'] = width

        # indicate that the request was a success
        data["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))

    #app.run(debug=True)
    app.run(host='0.0.0.0', port = 5001)
