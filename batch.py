import keras.utils
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import SeparableConv2D, DepthwiseConv2D
from sklearn.utils import shuffle
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras import backend as K
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing, model_selection
from sklearn.utils import class_weight
from EegNet import EEGNet
import datetime
import itertools
import sklearn.metrics
import io
def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred_raw = Default.predict(X_test)
  test_pred = np.argmax(test_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(np.argmax(Y_test,axis=1), test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names = ['0','1','2','3','4'])
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("epoch_confusion_matrix", cm_image, step=epoch)

# Define the per-epoch callback.
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

#This is here for debugging purposes
DATA = r"DATA HERE"
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 2    
print("Reading data")

df_main = pd.read_csv(DATA)
print(f"Data read! Shape: {df_main.shape}")

data = df_main


data = data.drop(columns=['C1','Unnamed: 0','Time','C2','CCP2h','CCP4h','FCC2h']) #Removing these columns changes percentages as per our paper
print("Number of columns:", len(data.columns))
print("Columns:", data.columns)
print(data["label"].unique(), "\n")
print(len(data["label"].unique()), "\n")

np.random.seed(42)
#One-hot encoding
data.replace(
    {'label': 
   {
        776: 0,
        777: 1,
        779: 2,
        925: 3,
        926: 4
   } 
    },
    inplace=True,
)

print("After replacement")
(print(data["label"].unique()))
Y = data['label']
X = data.drop(columns=['label'])


print("Data shape after adjustment:", X.shape)
print("Y shape: ", Y.shape)
chan = len(data.columns) -1
classes = 5
X_train, X_test_val,Y_train,Y_test_val  = model_selection.train_test_split(X,Y, test_size =0.2,shuffle=True,random_state = 42)
X_test,X_val,Y_test, Y_val = model_selection.train_test_split(X_test_val,Y_test_val, test_size=0.5, shuffle=True,random_state= 42)
#Reshapes data, unless you are using multiple subjects this should remain unchanged
X_train = np.asarray(X_train).astype(np.float32).reshape(-1,chan,1,1)
X_test = np.asarray(X_test).astype(np.float32).reshape(-1,chan,1,1)
X_val = np.asarray(X_val).astype(np.float32).reshape(-1,chan,1,1)
Y_train = keras.utils.to_categorical(Y_train)
Y_val = keras.utils.to_categorical(Y_val)
Y_test = keras.utils.to_categorical(Y_test)
print("X train shape:", X_train.shape)
print("Y train shape:", Y_train.shape)
print("X val shape: ", X_val.shape)
print("Y val shape: ", Y_val.shape)
print("X test shape", X_test.shape)
print("Y test shape", Y_test.shape)





#Convert the Data into ML-readable format
TRAIN = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
TEST = tf.data.Dataset.from_tensor_slices((X_test,Y_test))
VAL = tf.data.Dataset.from_tensor_slices((X_val,Y_val))
#Shuffle according to batch size to prevent overfitting while also allowing for the temporal relations to be preserved
TRAIN = TRAIN.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
TEST = TEST.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
VAL = VAL.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)



#Debugging purposes
tf.debugging.enable_check_numerics()
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
#Fine tune these parameters for higher results.  We empirically found these parameters to yeild the best results for our data
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "~/Classifers/checkpoint.h5", save_best_only=True, monitor="loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.000001,
    ),tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),cm_callback

]

print("Default")
Default = EEGNet(nb_classes= classes,Chans = chan,Samples=1)
class_weights_default = {0:1, 1:1, 2:1, 3:1,4:1} #change this
Default.summary()
Default.compile(loss = 'categorical_crossentropy', optimizer = 'adam',   metrics = ['accuracy'])
fittedModel = Default.fit((TRAIN), epochs = 100, 
                        verbose = 2, validation_data=VAL,
                        callbacks= callbacks, class_weight = class_weights_default)


import matplotlib.pyplot as plt

history = fittedModel.history

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()
