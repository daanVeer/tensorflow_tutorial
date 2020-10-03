import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__) # 2.3

'''
this tutorial is the image classification using MNIST dataset.
'''

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# explore the data

'''
MNIST image dataset has 28*28 dimension.
'''
print(train_images.shape) # (60000, 28, 28)
print(len(train_labels)) # 60000

print(test_labels.shape) # (10000, 28, 28)
print(len(test_labels)) # 10000

# pre process the data

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.savefig('./results_plot/Basic_Image_Classification_1.png')

# to normalize the image data in range 0 to 1
train_images = train_images/255.0
test_images = test_images/255.0

plt.figure(figsize = (10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i], cmap = plt.cm.binary)
	plt.xlabel(class_names[train_labels[i]])

plt.savefig('./results_plot/Basic_Image_Classification_2.png')

# build the model

# setup the layers with keras
model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28, 28)), # input shape is (28, 28)
	keras.layers.Dense(128, activation = 'relu'), # just hidden unit size
	keras.layers.Dense(10)]) # output class = 10 (0~9)

# compile the model
model.compile(optimizer = 'adam',
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
	metrics = ['accuracy'])

# train the model
'''
train steps:

1. Feed the training data to the model. 
   In this example, the training data is in the train_images and train_labels arrays.

2. The model learns to associate images and labels.

3. You ask the model to makek predictions about a test set-in this example, the test_images array.

4. Verify that the predictions match the labels from the test_labels array. 
'''

model.fit(train_images, train_labels, epochs = 10)

# evaluate accuracy

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print('\n Test accuracy: ', test_acc)


# make predictions

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

print('prediction value: ', np.argmax(predictions[0]), ' real test value: ', test_labels[0])


# for graph

def plot_image(i, predictions_array, true_label, img):
	true_label, img = true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	
	plt.imshow(img, cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
				100*np.max(predictions_array),
				class_names[true_label]),
				color=color)

def plot_value_array(i, predictions_array, true_label):
	true_label = true_label[i]
	plt.grid(False)
	plt.xticks(range(10))
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

# verify predictions
i = 0 # example number 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.savefig('./results_plot/Basic_Image_Classification_3.png')

i = 12 # example number 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.savefig('./results_plot/Basic_Image_Classification_4.png')


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
	plt.subplot(num_rows, 2*num_cols, 2*i+1)
	plot_image(i, predictions[i], test_labels, test_images)
	plt.subplot(num_rows, 2*num_cols, 2*i+2)
	plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.savefig('./results_plot/Basic_Image_Classification_5.png')



# use the trained model (pre-train model)

img = test_images[1]
print(img.shape) # (28, 28)

img = (np.expand_dims(img, 0)) # add the image to a batch where it's the only member.
print(img.shape) # (1, 28, 28)

predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation = 45)

print(np.argmax(predictions_single[0]))
