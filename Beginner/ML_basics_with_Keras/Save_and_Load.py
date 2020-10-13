import os
import tensorflow as tf
from tensorflow import keras

# dataset

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28*28)/255
test_images = test_images[:1000].reshape(-1, 28*28)/255

# model

def create_model():
	model = tf.keras.models.Sequential([
		keras.layers.Dense(512, activation = 'relu', input_shape= (784,)),
		keras.layers.Dropout(0.2),
		keras.layers.Dense(10)])
	
	model.compile(optimizer='adam', loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True),
		metrics = [tf.metrics.SparseCategoricalAccuracy()])

	return model

model = create_model()
print(model.summary())


# checkpoint callback usage
## tf.keras.callbacks.ModelCheckpoint : continually save the model

checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

## create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, 
			save_weights_only = True, verbose=1)

## train the model with the new callback
model.fit(train_images, train_labels, epochs = 10, 
		validation_data = (test_images, test_labels), callbacks = [cp_callback])

model = create_model()
loss, acc = model.evaluate(test_images, test_labels, verbose = 2)
print('untrained model, accuracy: {:5.2f}%'.format(100*acc))

## load weights
model.load_weights(checkpoint_path)

## re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose = 2)
print('restored model, accuracy: {:5.2f}%'.format(100*acc))


# chcekpoint callback options

## include the epoch
checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
chcekpoint_dir = os.path.dirname(checkpoint_path)

## create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
		verbose = 1, save_weights_only = True, period = 5)

model = create_model()

model.save_weights(checkpoint_path.format(epoch = 0))

## train the model with the new callback
model.fit(train_images, train_labels, epochs = 50, callbacks = [cp_callback], 
		validation_data = (test_images, test_labels), verbose = 0)


latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

## create a new model instance
model = create_model()

model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels, verbose = 2)
print('restored model, accuracy: {:5.2f}%'.format(100*acc))


# manually save weights

model.save_weights('./checkpoints/my_checkpoint')

model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels, verbose = 2)
print('restored model, accuracy: {5.2f}%'.format(100*acc))


# saved model format
model = create_model()
model.fit(train_images, train_labels, epochs = 5)

model.save('saved_model/my_model')

new_model = tf.keras.models.load_model('saved_model/my_model')
print(new_model.summary())

loss, acc = new_model.evaluate(test_images, test_labels, verbose = 2)
print('restored model, accuracy: {:5.2f}%'.format(100*acc))

print(new_model.predict(test_images).shape)

# HDF5 format
model = create_model()
model.fit(train_images, train_labels, epochs = 5)

model.save('my_model.h5')

new_model = tf.keras.models.load_model('my_model.h5')
print(new_model.summary())

loss, acc = new_model.evaluate(test_images, test_labels, verbose = 2)
print('restored model, accuracy: {:5.2f}%'.format(100*acc))



















