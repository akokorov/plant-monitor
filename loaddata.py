#construct data frame csv with image file names, startX, startY, endX, endY of ROI from supervise.ly json files

#create file list from folder
from os import listdir
from os.path import isfile, join

import json

import pandas as pd
import numpy as np

mypath = 'D:/ML/plant-monitor/jsonrect'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

datalist=[]
height = 480
width = 640

for f in onlyfiles :
    #print(f)
    with open('jsonrect/'+f) as json_file:
        data = json.load(json_file)
    if len(data['objects']) > 0 :
        rect = np.array(data['objects'][0]['points']['exterior'])
        rect = np.reshape(rect,-1)

        startX = rect[0]/width
        startY = rect[1]/height
        endX = rect[2]/width
        endY = rect[3]/height

        file_name = f.replace(".json", "")

        datalist.append([file_name, startX, startY, endX, endY])


df = pd.DataFrame(datalist,columns = ["file name",'startX','startY','endX','endY'])

#correct wrong labelling (startX > endX and startY > endY)
df_x = df.loc[df['startX']>df['endX']]
df_y = df.loc[df['startY']>df['endY']]
for i in df_x.index:
    df.loc[i, 'startX'] = df_x.loc[i, 'endX']
    df.loc[i, 'endX'] = df_x.loc[i, 'startX']
for i in df_y.index:
    df.loc[i, 'startY'] = df_y.loc[i, 'endY']
    df.loc[i, 'endY'] = df_y.loc[i, 'startY']

#save label data from supervisely as label_from_svl.csv
df.to_csv("label_from_svl.csv")

#append label data from supervisely to label data from boxlabel(from EAST and SSD)
#df=df.append(pd.read_csv('label_add.csv')[["file name",'startX','startY','endX','endY']])


#save appended label data as label_roi.csv
#df.to_csv('label_roi.csv')

#split label data to 0.9 train set and .01 test set
out = np.random.permutation(len(df))
data_shuf = df.iloc[out]

splt = int(np.floor(len(data_shuf)*0.90))

data_train = data_shuf.iloc[:splt]
data_test = data_shuf.iloc[splt:]
data_train.to_csv("data_train.csv")
data_test.to_csv("data_test.csv")


