# import the necessary packages
import requests
import numpy as np
import pandas as pd
import json
import time
from time import localtime, strftime
import traceback
import cv2


# initialize the Keras REST API endpoint URL
KERAS_REST_API_URL = "http://127.0.0.1:5001/predict"
#KERAS_REST_API_URL = "http://app02-stg.inwini.com/rvp/predict"
#KERAS_REST_API_URL = "https://amr-anomaly-ce6ez7jgkq-uc.a.run.app/predict"


# load the input dict and construct the payload for the request

data = pd.read_csv('label_from_svl.csv', index_col = 0)
i=100

for i in range(len(data)):
    try:
        image_path = data['file name'].iloc[i]
        image = cv2.imread('image/'+image_path)
        myfiles = {'file': open('image/'+image_path, 'rb')}
        start = time.time()
        r = requests.post(KERAS_REST_API_URL, files=myfiles).json()
        # ensure the request was successful
        if r["success"]:
            # loop over the predictions and display them
            end = time.time()
            print("[INFO] calculation at {} for sample number {} took {:.6f} seconds".format(strftime("%Y%m%d%H%M%S", localtime()),0,end - start))
            print('device ID =', r["device_id"])
            print('height =', r["height"])
            print('width = ', r["width"])
            print('JSON = ', r)
            print('==========================')

            startX = r['startX']
            startY = r['startY']
            endX = r['endX']
            endY = r['endY']
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            cv2.imshow("Text detection result", image)
            cv2.waitKey(0)

        # otherwise, the request failed
        else:
            print("Request failed")
        #time.sleep(5)

    except Exception as e:
        print("[INFO] request fail at {} for sample number {}".format(strftime("%Y%m%d%H%M%S", localtime()), 0))
        print('==========================')
        print(traceback.print_exc())
        time.sleep(1)
    #i += 1

