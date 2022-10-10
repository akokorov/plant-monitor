from keras.applications.nasnet import NASNetMobile
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten

# create the base pre-trained model
#base_model = VGG16(weights='imagenet', include_top=False,  input_shape=(240,320,3))

base_model = NASNetMobile(weights=None, include_top=False, input_shape=(240,320,3), pooling='max')

# add a global spatial average pooling layer
x = base_model.output
#x = Flatten()(x)
#x = GlobalAveragePooling2D()(x)
#x = Dropout(0.2)(x)
# let's add a fully-connected layer
#x = Dense(256, activation='relu')(x)
#x = Dropout(0.2)(x)
#x = Dense(128, activation='relu')(x)
# and a logistic layer
#x = Dropout(0.2)(x)
x = Dropout(0.25)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='sigmoid')(x)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
#    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)

model.compile(loss='mean_squared_error', optimizer='rmsprop') # Use categorical_crossentropy as the number of classes are more than one.
#adam = optimizers.Adam(lr=0.00005, decay=0)
#model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["categorical_accuracy"])
#model.compile(optimizers.rmsprop(lr=0.0001,loss='categorical_crossentropy',metrics=["categorical_accuracy"]))
#from keras.optimizers import SGD
#model.compile(optimizer=SGD(lr=0.00005, momentum=0.9), loss='categorical_crossentropy',metrics=["categorical_accuracy"])
model.summary()

model.save('my_model_roi_nasnet.h5')