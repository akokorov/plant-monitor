from keras_preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import load_model
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from time import localtime, strftime

df_train=pd.read_csv("data_train.csv")
datagen_train=ImageDataGenerator(
    rescale=1./255,)
    #brightness_range=[0.9,1.1])
#mean = np.array([123.68, 116.779, 103.939], dtype="float32")
#datagen_train.mean = mean
train_generator=datagen_train.flow_from_dataframe(dataframe=df_train, directory="D:\ML\plant-monitor\image", x_col="file name", y_col=['startX','startY','endX','endY'], class_mode ="other", target_size=(240,320), batch_size=64, shuffle=True)

df_valid=pd.read_csv("data_test.csv")
datagen_valid=ImageDataGenerator(rescale=1./255)
#datagen_valid.mean = mean
valid_generator=datagen_valid.flow_from_dataframe(dataframe=df_valid, directory="D:\ML\plant-monitor\image", x_col="file name", y_col=['startX','startY','endX','endY'], class_mode ="other", target_size=(240,320), batch_size=64)


#hist = np.array(pd.read_csv("hist2.csv")["loss"])



model=load_model('my_model_roi_nasnet_trained.h5',compile = False)

model.summary()

#def l2loss(y_true,y_pred):
#    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_true,y_pred))))
adam = optimizers.Adam(lr=0.00001)
model.compile(optimizer=adam,loss='mean_squared_error',metrics=['accuracy'])

#model.compile(loss='mean_squared_error', optimizer='rmsprop')

#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["categorical_accuracy"]) # Use categorical_crossentropy
#RMSprop = optimizers.RMSprop(lr=0.001, decay=0)
#model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=["categorical_accuracy"])

#from keras.optimizers import SGD
#model.compile(optimizer=SGD(lr=0.00005, momentum=0.9), loss='categorical_crossentropy',metrics=["categorical_accuracy"])



STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=30000,

                    callbacks = [ReduceLROnPlateau(patience=1000)])
#class_weight=class_weights,
#save model

model.save('my_model_roi_nasnet_trained.h5')
#hist = np.append(hist, history.history["loss"], axis=0)
#print(hist)
print("finished, model saved")
history = pd.DataFrame(history.history)

filename = 'history_'+strftime("%Y%m%d%H%M%S", localtime())
#filename = 'history_20200421164011.csv'

with open('history/'+filename+'.csv', 'a') as f:
    history = history.loc[pd.notnull(history.loss)] #remove NaN
    history.to_csv(f, header=True)

#plot loss

import matplotlib.pyplot as plt
df = history.loc[pd.notnull(history.loss)].iloc[:] #remove NaN

N=len(df["loss"])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), df["loss"], label="train_loss")
plt.plot(np.arange(0, N), df["val_loss"], label="val_loss")
plt.legend(loc="upper right")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.savefig('history/'+filename+'.png')
plt.show()