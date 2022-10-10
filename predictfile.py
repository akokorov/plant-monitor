from keras.models import load_model

import pandas as pd
import numpy as np
import cv2
import time


model=load_model('my_model_roi_nasnet_trained.h5', compile=False)

#load and predict test image

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



data_test=pd.read_csv("label_from_svl.csv")[:]
i=0
h = 480
w = 640
for i in range(len(data_test)):
    try :
        #image_path=input('enter file name : ')
        #if image_path=='q' :
        #    break
        image_path=data_test['file name'].iloc[i]
        #image_path='20190618190139.jpg'

        image = cv2.imread('image/'+image_path)
        start = time.time()
        box = roi_lenet(image)
        end = time.time()
        print("[INFO] text detection took {:.6f} seconds".format(end - start))

        print(box)

        startX, startY, endX, endY = box

        id = '00000001'
        height = endY-startY
        width = endX-startX


        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        cv2.rectangle(image, (520, 400), (640, 480), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, 'ID = {}'.format(id), (530, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0),
                    thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(image, 'Height = {}'.format(height), (530, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0),
                    thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(image, 'Width = {}'.format(width), (530, 460), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


        cv2.imshow("Plant detection result", image)
        cv2.waitKey(10)



    except FileNotFoundError:
        print('file not found')

    except Exception as e:
        print(e)