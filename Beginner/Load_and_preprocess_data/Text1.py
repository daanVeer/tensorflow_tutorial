import collections
import pathlib
import re
import string

import tensorflow as tf
from tensorflow.keras import layers, losses, preprocessing, utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text

# prediction the tag

data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
dataset = utils.get_file(
	'stack_overflow_16k.tar.gz',
	data_url, untar = True,
	cache_dir = 'stack_overflow', cache_subdir = '')

## dataset
dataset_dir = pathlib.Path(dataset).parent

print(list(dataset_dir.iterdir()))

train_dir = dataset_dir/'train'
print(list(train_dir.iterdir()))

sample_file = train_dir/'python/1755.txt'
with open(sample_file) as f:
	print(f.read())

batch_size = 32
seed = 119

## data split
raw_train_ds = preprocessing.text_dataset_from_directory(
	train_dir, batch_size = batch_size, 
	validation_split = 0.2, subset = 'training', seed = seed)


for text_batch, label_batch in raw_train_ds.take(1):
	for i in range(10):
		print('question: ', text_batch.numpy()[i][:100], '...')
		print('label: ', label_batch.numpy()[i])

for i, label in enumerate(raw_train_ds.class_names):
	print('label', i, 'corresponds to', label)

raw_val_ds = preprocessing.text_dataset_from_directory(
	train_dir, batch_size = batch_size,
	validation_split = 0.2, subset = 'validation', seed = seed)

test_dir = dataset_dir/'test'
raw_test_ds = preprocessing.text_dataset_from_directory(
	test_dir, batch_size = batch_size)

## prepare dataset

vocab_size = 10000
binary_vectorize_layer = TextVectorization(
	max_tokens = vocab_size, output_mode = 'binary')

max_sequence_length = 250
int_vectorize_layer = TextVectorization(
	max_tokens = vocab_size, output_mode = 'int', 
	output_sequence_length = max_sequence_length)



train_text = raw_train_ds.map(lambda text, labels: text)
binary_vectorize_layer.adapt(train_text)
int_vectorize_layer.adapt(train_text)


def binary_vectorize_text(text, label):
	text = tf.expand_dims(text, -1)
	return binary_vectorize_layer(text), label

def int_vectorize_text(text, label):
	text = tf.expand_dims(text, -1)
	return int_vectorize_layer(text), label


text_batch, label_batch = next(iter(raw_train_ds))
first_question, first_label = text_batch[0], label_batch[0]
print('question: ', first_question)
print('label: ', first_label)

print('binary vectorized question: ', binary_vectorize_text(first_question, first_label)[0])
print('int vectorized question: ', int_vectorize_text(first_question, first_label)[0])

print("1289 -> ", int_vectorize_layer.get_vocabulary()[1289])
print("313 -> ", int_vectorize_layer.get_vocabulary()[313])
print("Vocabulary size: {}".format(len(int_vectorize_layer.get_vocabulary())))

binary_train_ds = raw_train_ds.map(binary_vectorize_text)
binary_val_ds = raw_val_ds.map(binary_vectorize_text)
binary_test_ds = raw_test_ds.map(binary_vectorize_text)

int_train_ds = raw_train_ds.map(int_vectorize_text)
int_val_ds = raw_val_ds.map(int_vectorize_text)
int_test_ds = raw_test_ds.map(int_vectorize_text)

Autotune = tf.data.experimental.AUTOTUNE

def configure_dataset(dataset):
	return dataset.cache().prefetch(buffer_size = Autotune)

binary_train_ds = configure_dataset(binary_train_ds)
binary_val_ds = configure_dataset(binary_val_ds)
binary_test_ds = configure_dataset(binary_test_ds)

int_train_ds = configure_dataset(int_train_ds)
int_val_ds = configure_dataset(int_val_ds)
int_test_ds = configure_dataset(int_test_ds)

## train model
binary_model = tf.keras.Sequential([layers.Dense(4)])
binary_model.compile(loss = losses.SparseCategoricalCrossentropy(from_logits = True), 
	optimizer = 'adam', metrics = ['accuracy'])

history = binary_model.fit(binary_train_ds, validation_data = binary_val_ds, epochs = 10)

def create_model(vocab_size, num_labels):
	model = tf.keras.Sequential([
		layers.Embedding(vocab_size, 64, mask_zero=True),
		layers.Conv1D(64, 5, padding = 'valid', activation = 'relu', strides = 2),
		layers.GlobalMaxPooling1D(),
		layers.Dense(num_labels)])
	return model

int_model = create_model(vocab_size = vocab_size + 1, num_labels = 4)
int_model.compile(loss = losses.SparseCategoricalCrossentropy(from_logits = True),
	optimizer = 'adam', metrics = ['accuracy'])

history = int_model.fit(int_train_ds, validation_data = int_val_ds, epochs = 5)

print('linear model on binary vectorized data: ')
print(binary_model.summary())

print('ConvNet model on int vectorized data: ')
print(int_model.summary())

binary_loss, binary_accuracy = binary_model.evaluate(binary_test_ds)
int_loss, int_accuracy = int_model.evaluate(int_test_ds)

print('binary model accuray: {:2.2%}'.format(binary_accuracy))
print('int model accuracy: {:2.2%}'.format(int_accuracy))

## export model

export_model = tf.keras.Sequential([
	binary_vectorize_layer, binary_model, 
	layers.Activation('sigmoid')])

export_model.compile(
	loss = losses.SparseCategoricalCrossentropy(from_logits = False),
	optimizer = 'adam', metrics = ['accuracy'])


loss, accuracy = export_model.evaluate(raw_test_ds)
print('accuracy: {:2.2%}'.format(binary_accuracy))

def get_string_labels(predicted_scores_batch):
	predicted_int_labels = tf.argmax(predicted_scores_batch, axis = 1)
	predicted_labels = tf.gather(raw_train_ds.class_names, predicted_int_labels)
	return predicted_labels


inputs = ["how do I extract keys from a dict into a list?",  # python
	"debug public static void main(string[] args) {...}"]  # java
predicted_scores = export_model.predict(inputs)
predicted_labels = get_string_labels(predicted_scores)

for input, label in zip(inputs, predicted_labels):
	print("question: ", input)
	print("predicted label: ", label.numpy())



