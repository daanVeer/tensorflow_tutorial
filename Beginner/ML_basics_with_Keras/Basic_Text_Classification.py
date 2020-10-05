import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# data download
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url, untar = True, cache_dir = '.', cache_subdir = 'data')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

os.listdir(dataset_dir)


train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir) 

sample_file = os.path.join(train_dir, 'pos/1181_9.txt')

with open(sample_file) as f:
	print(f.read())

# load dataset

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# make validation dataset
'''
train dataset: 25,000 
> train: 20,000 (0.8)
> validation: 5,000 (0.2)
'''
batch_size = 32

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
	'data/aclImdb/train',
	batch_size = batch_size,
	validation_split = 0.2, 
	subset = 'training',
	seed = 42)

# for example
for text_batch, label_batch in raw_train_ds.take(1):
	for i in range(3):
		print('review: ', text_batch.numpy()[i])
		print('label: ', label_batch.numpy()[i])

print('label 0 corresponds to ', raw_train_ds.class_names[0])
print('label 1 corresponds to ', raw_train_ds.class_names[1])

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
	'data/aclImdb/train',
	batch_size = batch_size,
	validation_split = 0.2,
	subset = 'validation',
	seed = 42)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
	'data/aclImdb/test',
	batch_size = batch_size)


# prepare the dataset for training
# normalization or regex or vectorization etc.

def custom_standardization(input_data): # regex the text data
	lowercase = tf.strings.lower(input_data)
	stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
	return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation),'')

max_features = 10000
sequence_length = 250

vectorize_layer = TextVectorization(
	standardize = custom_standardization,
	max_tokens = max_features,
	output_mode = 'int',
	output_sequence_length = sequence_length)

# make a text only dataset (just x features)
train_text = raw_train_ds.map(lambda x, y:x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
	text = tf.expand_dims(text, -1)
	return vectorize_layer(text), label

# retrieve a batch (32)
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print('review ', first_review)
print('label ', raw_train_ds.class_names[first_label])
print('vectorized review ', vectorize_text(first_review, first_label))

# for example
print('1287 > ', vectorize_layer.get_vocabulary()[1287])
print('313  > ', vectorize_layer.get_vocabulary()[313])
print('vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# configure the dataset for performance
# .cache() keeps data in memory after it's loaded off disk.
# .prefetch() overlaps data preprocessing and model execution while training.

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size = AUTOTUNE)

# create the model

embedding_dim = 16 

model = tf.keras.Sequential([
	layers.Embedding(max_features + 1, embedding_dim), # embedding to encode the text into integer. 
	# embedding dimensions are (batch, sequence, embedding)
	layers.Dropout(0.2), # reguralization
	layers.GlobalAveragePooling1D(), # average pooling
	layers.Dropout(0.2), # reguralization
	layers.Dense(1)]) # a single output node.

print(model.summary())

model.compile(loss = losses.BinaryCrossentropy(from_logits = True), optimizer = 'adam', metrics = tf.metrics.BinaryAccuracy(threshold=0.0))

# train the model

epochs = 10
history = model.fit(train_ds, validation_data = val_ds, epochs = epochs)

# evaluate

loss, accuracy = model.evaluate(test_ds)

print('loss: ', loss)
print('accuracy: ', accuracy)

# create a plot of accuracy and loss over time

history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label = 'Training loss') # blue dot
plt.plot(epochs, val_loss, 'b', label = 'Validation loss') # solid blue line
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('./results_plot/Basic_Text_Classification_1.png')

plt.plot(epochs, acc, 'ro', label = 'Training acc') # red dot
plt.plot(epochs, val_acc, 'r', label = 'Valindation acc') # solid red line
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc = 'lower right')
plt.savefig('./results_plot/Basic_Text_Classification_2.png')


# export the model

export_model = tf.keras.Sequential([vectorize_layer, # our vectorizer
	model, # pre-train model
	layers.Activation('sigmoid')]) # output activation function

export_model.compile(loss = losses.BinaryCrossentropy(from_logits=False), optimizer = 'adam', metrics = ['accuracy'])

# test it with 'raw test ds'
loss, accuracy = export_model.evaluate(raw_test_ds)


# inference on new data
examples = [
	'The movie was great!',
	'The movie was okay.',
	'The movie was terrible...']

predict = export_model.predict(examples) # test new data
print(predict)












































