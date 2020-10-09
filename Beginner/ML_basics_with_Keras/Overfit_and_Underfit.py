import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors = True)

# dataset

gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
features = 28

ds = tf.data.experimental.CsvDataset(gz, [float(),]*(features+1), compression_type='GZIP')

def pack_row(*row):
	label = row[0]
	features = tf.stack(row[1:], 1)
	return features, label

packed_ds = ds.batch(10000).map(pack_row).unbatch()

for features, label in packed_ds.batch(1000).take(1):
	print(features[0])
	plt.hist(features.numpy().flatten(), bins = 101)

plt.savefig('./results_plot/Overfit_and_Underfit_1.png')
plt.clf()

n_validation = int(1e3)
n_train = int(1e4)
buffer_size = int(1e4)
batch_size = 500
steps_per_epoch = n_train//batch_size

validation_ds = packed_ds.take(n_validation).cache()
train_ds = packed_ds.skip(n_validation).take(n_train).cache()
print(train_ds)

validation_ds = validation_ds.batch(batch_size)
train_ds = train_ds.shuffle(buffer_size).repeat().batch(batch_size)

# demonstrate overfitting

'''
데이터의 용량이 클 때, 즉 다차원 공간에 대하여 많은 feature들을 처리하고자 할 때 
훈련 데이터에 너무 적합하여 새로운 샘플 데이터에 대하여 처리하지 못하는 경우를 overfitting이라고 한다.
기계학습에서는 훈련 데이터의 성능과 새로운 테스트 데이터의 성능 차이가 적을수록 일반화 능력이 크다고 한다.
만약, overfitting이라고 한다면 훈련 데이터의 성능은 크고 새로운 테스트 데이터의 성능을 적게 나타날 것이다. 
'''

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps = steps_per_epoch*1000, 
		decay_rate = 1, staircase = False)

def get_optimizer():
	return tf.keras.optimizers.Adam(lr_schedule)

step = np.linspace(0, 100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/steps_per_epoch, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.savefig('./results_plot/Overfit_and_Underfit_2.png')
plt.clf()


def get_callback(name):
	return [
		tfdocs.modeling.EpochDots(),
		tf.keras.callbacks.EarlyStopping(monitor = 'val_binary_crossentropy', patience = 200),
		tf.keras.callbacks.TEnsorBoard(logdir/name),]

def compile_and_fit(model, name, optimizer = None, max_epochs=10000):
	if optimizer is None:
		optimizer = get_optimizer()
	model.compile(optimizer = optimizer, loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
		metrics = [tf.keras.losses.BinaryCrossentropy(from_logits=True, name = 'binary_crossentropy'), 'accuracy'])
	
	print(model.summary())

	history = model.fit(train_ds, 
			steps_per_epoch = steps_per_epoch,
			epochs = max_epochs, validation_data = validatoin_ds,
			callbacks = get_callbacks(name), verbose = 0)
	return history


# tiny model

tiny_model = tf.keras.Sequential([
		layers.Dense(16, activation = 'elu', input_shape = (features,)),
		layers.Dense(1)])

size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std = 10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
plt.savefig('./results_plot/Overfit_and_Underfit_3.png')
plt.clf()

# small model

small_model = tf.keras.Sequential([
	layers.Dense(16, activation = 'elu', input_shape = (features,)),
	layers.Dense(16, activation = 'elu'),
	layers.Dense(1)])

size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

# medium model

medium_model = tf.keras.Sequential([
	layers.Dense(64, activation = 'elu', input_shape = (features,)),
	layers.Dense(64, activation = 'elu'),
	layers.Dense(64, activation = 'elu'),
	layers.Dense(1)])

size_histories['Medium'] = compile_and_fit(medium_model, 'sizes/Medium')

# large model

large_model = tf.keras.Sequential([
	layers.Dense(512, activation = 'elu', input_shape = (features,)),
	layers.Dense(512, activatoin = 'elu'),
	layers.Dense(512, activation = 'elu'),
	layers.Dense(512, activation = 'elu'),
	layers.Dense(1)])

size_histories['large'] = compile_and_fit(large_model, 'sizes/large')


# plot the training and validation losses

plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel('epoch [log scale]')
plt.savefit('./results_plot/Overfit_and_Underfit_4.png')
plt.clf()

#%load_ext tensorboard
#%tensorboard --logdir {logdir}/sizes

display.IFrame(
    src="https://tensorboard.dev/experiment/vW7jmmF9TmKmy3rbheMQpw/#scalars&_smoothingWeight=0.97",
    width="100%", height="800px")

# prevent overfitting

shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')

regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']

# solution 1
# weight regularization

l2_momdel = tf.keras.Sequential([
	layers.Dense(512, activation = 'elu', kernel_regularizer = regularizers.l2(0.001), input_shape = (features,)),
	layers.Dense(512, activation = 'elu', kernel_regularizer = regularizers.l2(0.001)),
	layers.Dense(512, activation = 'elu', kerner_regularizer = regularizers.l2(0.001)),
	layers.Dense(512, activation = 'elu', kerner_regularizer = regularizers.l2(0.001)),
	layers.Dense(1)])

regularizer_histories['l2'] = compile_and_fit(l2_odel, 'regularizers/l2')

plotter.plot(regularizer_histores)
plt.ylim([0.5, 0.7])
plt.savefit('./results_plot/Overfit_and_Underfit_5.png')
plt.clf()

results = l2_model(features)
regularization_loss = tf.add_n(l2_model.losses)

# solution 2
# add dropout

dropout_model = tf.keras.Sequential([
	layers.Dense(512, activation = 'elu', input_shape=(features,)),
	layers.Dropout(0.5),
	layers.Dense(512, activation = 'elu'),
	layers.Dropout(0.5),
	layers.Dense(512, activation = 'elu'),
	layers.Dropout(0.5),
	layers.Dense(512, activation = 'elu'),
	layers.Dropout(0.5),
	layers.Dense(1)])

regularizer_histories['dropout'] = compile_and_fit(dropout_model, 'regularizers/dropout')

plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.savefit('./results_plot/Overfit_and_Underfit_6.png')
plt.clf()

# combination l2+dropout

combined_model = tf.keras.Sequential([
	layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu', input_shape=(FEATURES,)),
	layers.Dropout(0.5),
	layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'),
	layers.Dropout(0.5),	
	layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'),
	layers.Dropout(0.5),
	layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'),
	layers.Dropout(0.5),
	layers.Dense(1)])

regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")

plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.savefit('./results_plot/Overfit_and_Underfit_7.png')
plt.clf()

#%tensorboard --logdir {logdir}/regularizers

display.IFrame(
    src="https://tensorboard.dev/experiment/fGInKDo8TXes1z7HQku9mw/#scalars&_smoothingWeight=0.97",
    width = "100%",
    height="800px")

