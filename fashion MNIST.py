import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout, Flatten,  Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard,  ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import os

# Check GPU
import tensorflow as tf
tf.config.list_physical_devices('GPU')


# Variables
epochs = 2000
num_classes = 10
batch_size = 128
img_rows, img_cols = 28, 28

# Grab the data from the keras repository

mnist_data = fashion_mnist.load_data()
x = mnist_data[0][0]
y = mnist_data[0][1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

# Process the date into the right tensor shape.  This is a good practice, but
# usually tensorflow uses channels last (the 'else' here)

if K.image_data_format() == "channels first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
#
#  Cast to a 32 bit float and then scale so the value is a float between 0 and 1
    
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

#
# Convert Class Vector to Binary Class Matrices (one-hot encoding).
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test.shape)

#
# Function to decode one-hot encoding later on when we want to evaluate performance.
def decode_one_hot(y):
    y_classes = [np.argmax(yi, axis=None, out=None) for yi in y]
    return y_classes

#
#  Set up our Image Augmentation Data Generator
#
datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.3,
            zoom_range=(0.9, 1.1),
            horizontal_flip=True,
            vertical_flip=False, 
            fill_mode='constant',
            cval=0)

datagen.fit(x_train)


lam = 0 #1e-5

# Build model
model = Sequential()

model.add(InputLayer(input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
#model.add(BatchNormalization())
model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
#model.add(BatchNormalization())
model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
#model.add(Dropout(0.2))
model.add(MaxPooling2D((2,2)))


#model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
#model.add(BatchNormalization())
#model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
#model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
#model.add(Dropout(0.2))
model.add(MaxPooling2D((2,2)))


#model.add(BatchNormalization())
#model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
#model.add(BatchNormalization())
#model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
#model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
#model.add(Dropout(0.2))
model.add(MaxPooling2D((2,2)))


#model.add(BatchNormalization())
##model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
#model.add(BatchNormalization())
#model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
#model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
#model.add(Dropout(0.2))
model.add(MaxPooling2D((2,2)))

# model.add(BatchNormalization())
# model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
# model.add(BatchNormalization())
# model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
# model.add(BatchNormalization())
# model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(lam)))
# # #model.add(Dropout(0.2))


model.add(Flatten())
model.add(BatchNormalization())
#model.add(Dense(1024, activation='relu'))   
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))   
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))   
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))   
# model.add(Dropout(0.2))


model.add(Dense(num_classes, activation="softmax"))
model.add(Dense(num_classes, activation="softmax"))

# Having trouble getting tensorboard callback to work well with EarlyStopping callback...

my_callbacks = [ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=500,
        min_lr=0.00000001),
    EarlyStopping(monitor = 'val_loss', patience=50, min_delta=0.00001, mode='min', restore_best_weights=True),
    ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')]

model.summary()



# Flag to determine whether we use Keras' Image augmentation data generator
augmentation = True

# Compile the model so we can fit it.
model.compile(loss=keras.losses.categorical_crossentropy, 
              optimizer=keras.optimizers.adam(), metrics=['accuracy'])


hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                           steps_per_epoch=len(x_train) / batch_size, validation_data=(x_test, y_test),
                           epochs=epochs, verbose=1, callbacks=my_callbacks)


score = model.evaluate(x_test, y_test)

#
# Predict on the test data and pass to metrics function
yhat = model.predict_classes(x_test).astype(int)
y_dec = decode_one_hot(y_test)


print(metrics.classification_report(y_dec, yhat))
print("Testing Loss:", score[0])
print("Testing Accuracy:", score[1])


#model.summary()

#plot
epoch_list = list(range(1, len(hist.history['accuracy']) + 1))
plt.plot(epoch_list, hist.history['accuracy'], epoch_list, hist.history['val_accuracy'])
plt.legend(("Training Accuracy", "Validation Accuracy"))
plt.show()

plt.plot(epoch_list, hist.history['loss'], epoch_list, hist.history['val_loss'])
plt.legend(("Training Loss", "Validation Loss"))
plt.show()