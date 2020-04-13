kimport tensorflow as tf
# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import math
import numpy as np
import matplotlib.pyplot as plt


#Enable Logging
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)



#Load Data
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)


#Defining Training and Testing data
train_dataset, test_dataset = dataset['train'], dataset['test']

#Defining Labels
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#              'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

#Defining Kashmiri Labels
class_names = ['T-shirt/top', 'Yazaar', 'Chogi', 'Ferakh', 'Coat',
               'Chapin',      'Kameez',   'Booth',  'Beag',   'Long booth']



#Printing number of samples in Training and testing data dets
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples


print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

#print(repr(train_dataset.__dict__))
print(test_dataset.take(1))


#=================================================================
#Normalizing Images (mapping 0-255 range to 0-1)
def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)


# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()

#=============================================================================
# Take a single image, and remove the color dimension by reshaping
for image, label in test_dataset.take(1):
  break
image = image.numpy().reshape((28,28))


# Plot the first image from the input dataset
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.title('{}'.format(class_names[label]))
plt.show()


#===========================================================================
#Plotting first 16 images from Test Data

plt.figure()
i = 0
for (image, label) in test_dataset.take(16):

    image = image.numpy().reshape((28,28))
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1

plt.show()

#===========================================================================


#========================================================================
#Building Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(28, 28, 1)), 
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10)
])

#Compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)


# Training
model.fit(train_dataset, epochs=12, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

#Accuracy
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)



#Saving the model
#Use this model to make predictions insteal of creating model again and again
model.save("my_model.h5")
print("Model Saved !")

#=================================================================
#Plotting Accuracy and Loss
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#==========================================================================

#Now making predctions here
for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)


#Predictions are done foe first batch (32 images)
#print(predictions.shape) #Printing dimensions of predictions (Predictions of one batch i.e 32 images)


#Printing actual label of test data
print("Actual Image Label is: ")
print(class_names[test_labels[5]])


#Highest confidence of highest
pnum = np.argmax(predictions[5]) #getting the highest confidence index
print("Image predicted is: ")
print(class_names[pnum]) 




