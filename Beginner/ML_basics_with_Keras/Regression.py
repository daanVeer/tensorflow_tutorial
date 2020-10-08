import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision = 3, suppress = True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

# get dataset

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight','Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names = column_names, na_values = '?', comment = '\t', sep = ' ', skipinitialspace = True)

dataset = raw_dataset.copy()
print(dataset.tail())
print(dataset.isna().sum())

dataset = dataset.dropna() # remove the none data

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'}) # change the categorical column to one-hot

dataset = pd.get_dummies(dataset, prefix = '', prefix_sep = '')
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

sns_plot = sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind = 'kde')

sns_plot.savefig('./results_plot/Regression_1.png')


print(train_dataset.describe().transpose())

# split features from labels

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# normalization

print(train_dataset.describe().transpose()[['mean', 'std']])

## normalization layer

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
	print('first example: ', first)
	print()
	print('Normalized: ', normalizer(first).numpy())


# Linear Regression

horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = preprocessing.Normalization(input_shape=[1,])
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([
	horsepower_normalizer,
	layers.Dense(1)])

print(horsepower_model.summary())

print(horsepower_model.predict(horsepower[:10]))  # example 

horsepower_model.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.1), loss = 'mean_absolute_error')

history = horsepower_model.fit(train_features['Horsepower'], train_labels, 
	epochs = 100, verbose = 0, validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

def plot_loss(history, save_file):
	plt.plot(history.history['loss'], label = 'loss')
	plt.plot(history.history['val_loss'], label = 'val_loss')
	plt.ylim([0,10])
	plt.xlabel('epoch')
	plt.ylabel('error [MPG]')
	plt.legend()
	plt.grid(True)
	plt.savefig(save_file)
	plt.clf()

plot_loss(history, './results_plot/Regression_2.png')

test_results = {}
test_results['horsepower_model'] = horsepower_model.evaluate(test_features['Horsepower'], test_labels, verbose=0)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y, save_file):
	plt.scatter(train_features['Horsepower'], train_labels, label = 'Data')
	plt.plot(x, y, color = 'k', label = 'Predictions')
	plt.xlabel('Horsepower')
	plt.ylabel('MPG')
	plt.legend()
	plt.savefig(save_file)
	plt.clf()

plot_horsepower(x, y, './results_plot/Regression_3.png')


# multiple inputs # all features

linear_model = tf.keras.Sequential([normalizer, 
				layers.Dense(1)])

print(linear_model.predict(train_features[:10]))

print(linear_model.layers[1].kernel)


linear_model.compile(optimizer = tf.optimizers.Adam(learning_rate=0.1), loss = 'mean_absolute_error')

history = linear_model.fit(train_features, train_labels, epochs = 100, 
		verbose=0, validation_split = 0.2)


plot_loss(history, './results_plot/Regression_4.png')

test_results['linear_model'] = linear_model.evaluate(test_features, test_labels, verbose = 0)



# DNN regression

def build_and_compile_model(norm):
	model = keras.Sequential([norm,
			layers.Dense(64, activation = 'relu'),
			layers.Dense(64, activation = 'relu'),
			layers.Dense(1)])

	model.compile(loss='mean_absolute_error', optimizer = tf.keras.optimizers.Adam(0.001))
	return model

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
print(dnn_horsepower_model.summary())

history = dnn_horsepower_model.fit(train_features['Horsepower'], train_labels,
		validation_split = 0.2, verbose = 0, epochs=100)


plot_loss(history, './results_plot/Regression_5.png')


x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

plot_horsepower(x, y, './results_plot/Regression_6.png')

test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
			test_features['Horsepower'], test_labels, verbose=0)


# full model

dnn_model = build_and_compile_model(normalizer)
print(dnn_model.summary())


history = dnn_model.fit(train_features, train_labels, 
		validation_split = 0.2, verbose=0, epochs = 100)

plot_loss(history, './results_plot/Regression_7.png')

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

# performance

performance = pd.DataFrame(test_results, index = ['Mean absolute error [MPG]']).T
print(performance)

# make prediction

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect = 'equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('true values [MPG]')
plt.ylabel('predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.savefig('./results_plot/Regression_8.png')
plt.clf()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel('prediction error [MPG]')
plt.ylabel('Count')
plt.savefig('./results_plot/Regression_9.png')
plt.clf()

dnn_model.save('dnn_model')

# if I reload the model, it gives idential output

reloaded = tf.keras.models.load_model('dnn_model')
test_results['reload'] = reloaded.evaluate(test_features, test_labels, verbose=0)

reload_predict = pd.DataFrame(test_results, index = ['Mean absolute error [MPG]']).T
