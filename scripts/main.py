from keras.models import Sequential, load_model, Model
from keras.layers import Input, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D, \
		ZeroPadding2D, UpSampling2D, UpSampling1D
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD, Adadelta, Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, Callback
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
from os.path import dirname, abspath
from os import listdir, environ
import numpy as np
import h5py, pickle
from random import randint, choice, shuffle, sample
from sys import argv
from data import *
from predict import get_metrics
from bidaf import BiDAF
from bidaf_mod import BiDAF_mod
# from ema import ExponentialMovingAverage

tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
K.set_session(sess)

batch_size = [40, 40, 40, 40, 40, 40, 32]
noc, noq = 24, 1
# assert nop * noc * noq == batch_size
num_train, num_val = 80000, 10000

def init_model(preload=None, data=None):
	print 'Compiling model...'

	# model = RNet(hdim=75, word2vec_dim=EMBEDDING_DIM, data=data)
	# model = BiDAF(d=100, word2vec_dim=EMBEDDING_DIM, data=data)
	model = BiDAF_mod(d=100, word2vec_dim=EMBEDDING_DIM, data=data)

	# adadelta = Adadelta(lr=0.5)
	adam = Adam(beta_1=0.9, beta_2=0.9, clipnorm=5.0)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

	if preload:
		model.load_weights(preload)

	model.summary()

	print 'Done!\n'
	return model

def onehot(x, size):
	X = np.zeros((len(x), size), dtype=np.float32)
	for i, xx in enumerate(x):
		X[i][xx] = 1.
	return X

def multi_ans_onehot(x, size):
	X = np.zeros((3, len(x), size), dtype=np.float32)
	for i, xx in enumerate(x):
		for j, pos in enumerate(xx):
			X[j][i][pos] = 1.
	return X

def indices_generator(num_samples):
	idx = np.arange(num_samples)
	batch_indices = [idx[i:min(i + noc, num_samples)] for i in xrange(0, num_samples, noc)]

	for batch in batch_indices: yield batch

def DataGen(X, y, val=False, test=False):
	contexts, questions = X
	ans_starts, ans_ends = y

	num_samples = len(contexts)
	# contexts, questions = contexts[:500], questions[:500]
	# ans_starts, ans_ends = ans_starts[:500], ans_ends[:500]

	c_bins = [[] for _ in xrange(7)]
	q_bins = [[] for _ in xrange(7)]
	as_bins = [[] for _ in xrange(7)]
	ae_bins = [[] for _ in xrange(7)]

	def put(c, q, anss, anse, at):
		c_bins[at].append(c)
		q_bins[at].append(q)
		as_bins[at].append(anss)
		ae_bins[at].append(anse)

	indices = np.random.permutation(num_samples)
	contexts, questions = contexts[indices], questions[indices]
	ans_starts, ans_ends = ans_starts[indices], ans_ends[indices]
	index_gen = indices_generator(num_samples)

	while True:
		
		try: indices = index_gen.next()
		except StopIteration:
			if test:
				for i in xrange(len(c_bins)):
					if len(c_bins[i]) > 0:
						cur_c = c_bins[i]
						cur_q = q_bins[i]
						cur_as = as_bins[i]
						cur_ae = ae_bins[i]

						cur_c = pad_sequences(cur_c, maxlen=len(max(cur_c, key=len)), padding='post')
						cur_q = pad_sequences(cur_q, maxlen=len(max(cur_q, key=len)), padding='post')

						if test or val:
							cur_as = multi_ans_onehot(cur_as, size=cur_c.shape[1])
							cur_ae = multi_ans_onehot(cur_ae, size=cur_c.shape[1])

						else:
							cur_as = onehot(cur_as, size=cur_c.shape[1])
							cur_ae = onehot(cur_ae, size=cur_c.shape[1])

						yield [cur_c, cur_q], [cur_as, cur_ae]
				return

			else:
				indices = np.random.permutation(num_samples)
				contexts, questions = contexts[indices], questions[indices]
				ans_starts, ans_ends = ans_starts[indices], ans_ends[indices]
				index_gen = indices_generator(num_samples)

				continue

		cur_c, cur_q, cur_as, cur_ae = (contexts[indices], questions[indices], 
										ans_starts[indices], ans_ends[indices])

		for c, q, anss, anse in zip(cur_c, cur_q, cur_as, cur_ae):
			put(c, q, anss, anse, at=min(6, len(c) // 50))

		for i in xrange(len(c_bins)):
			if len(c_bins[i]) >= batch_size[i]:
				idx = len(c_bins[i]) - batch_size[i]

				cur_c = c_bins[i][idx:]
				cur_q = q_bins[i][idx:]
				cur_as = as_bins[i][idx:]
				cur_ae = ae_bins[i][idx:]

				del c_bins[i][idx:], q_bins[i][idx:], as_bins[i][idx:], ae_bins[i][idx:]

				cur_c = pad_sequences(cur_c, maxlen=len(max(cur_c, key=len)), padding='post')
				cur_q = pad_sequences(cur_q, maxlen=len(max(cur_q, key=len)), padding='post')

				# print cur_c.shape

				if test or val:
					cur_as = multi_ans_onehot(cur_as, size=cur_c.shape[1])
					cur_ae = multi_ans_onehot(cur_ae, size=cur_c.shape[1])

				else:
					cur_as = onehot(cur_as, size=cur_c.shape[1])
					cur_ae = onehot(cur_ae, size=cur_c.shape[1])

				yield [cur_c, cur_q], [cur_as, cur_ae]

class ValCheckPoint(Callback):
	def __init__(self, gen, test_data, to_file='bestval.h5'):
		self.gen = gen
		self.prev_loss = np.inf
		self.fname = to_file
		self.num_val_steps = num_val // batch_size[2]
		self.test_x, self.test_y = test_data
		self.test_steps = 269
		super(ValCheckPoint, self).__init__()
	
	def multi_ans_logloss(self, pred, act):
		epsilon = 1e-15
		pred = np.clip(pred, epsilon, 1 - epsilon)

		ll = np.sum(act * np.log(pred), axis=2)
		return -ll

	def tile_predictions(self, preds):
		preds = np.tile(preds.reshape(1, preds.shape[0], preds.shape[1]), [3, 1, 1])
		return preds 

	def on_epoch_end(self, epoch, logs={}):
		as_loss, ae_loss, as_acc, ae_acc = [], [], [], []

		for _ in xrange(self.num_val_steps):
			X, y_true = self.gen.next()
			as_preds, ae_preds = self.model.predict_on_batch(X)

			as_preds, ae_preds = self.tile_predictions(as_preds), self.tile_predictions(ae_preds)

			as_losses = self.multi_ans_logloss(as_preds, y_true[0])
			ae_losses = self.multi_ans_logloss(ae_preds, y_true[1])

			as_loss.append(np.mean(np.min(as_losses, axis=0)))
			ae_loss.append(np.mean(np.min(ae_losses, axis=0)))

			as_labels = np.argmax(as_preds, axis=2)
			ae_labels = np.argmax(ae_preds, axis=2)

			as_correct = (as_labels == np.argmax(y_true[0], axis=2))
			ae_correct = (ae_labels == np.argmax(y_true[1], axis=2))

			as_correct = np.logical_or(np.logical_or(as_correct[0], as_correct[1]), as_correct[2])
			ae_correct = np.logical_or(np.logical_or(ae_correct[0], ae_correct[1]), ae_correct[2])

			as_acc.append(np.sum(as_correct) / float(len(as_correct)))
			ae_acc.append(np.sum(ae_correct) / float(len(ae_correct)))

		as_loss = sum(as_loss) / len(as_loss)
		ae_loss = sum(ae_loss) / len(ae_loss)
		as_acc = sum(as_acc) / len(as_acc)
		ae_acc = sum(ae_acc) / len(ae_acc)

		total_loss = as_loss + ae_loss

		print '\nval_loss : {}\tval_as_loss : {}\tval_ae_loss : {}\tval_as_acc : {}\tval_ae_acc : {}\n'.format(total_loss,
																											as_loss,
																											ae_loss,
																											as_acc,
																											ae_acc)
		if total_loss < self.prev_loss:
			print 'val_loss improved from {} to {}; saving model to {}'.format(self.prev_loss, total_loss, self.fname)
			self.model.save_weights(self.fname)

		else:
			print 'val_loss did not improve...'

		self.prev_loss = total_loss

		# get F1 score and EM score
		test_gen = DataGen(self.test_x, self.test_y, test=True)
		get_metrics(self.model, test_gen, steps=self.test_steps)


def runner(preload, epochs):
	data = get_data()
	embedding_matrix = data['embedding_matrix']
	len_word_index = data['len_word_index']
	
	x_train, y_train = data['x_train'], data['y_pos_train']
	x_dev, y_dev = data['x_dev'], data['y_pos_dev']
	
	train_gen, val_gen = DataGen(x_train, y_train), DataGen(x_dev, y_dev, val=True)
	# return val_gen
	model = init_model(preload=preload, data=[embedding_matrix, len_word_index])

	print 'Model compiled.'

	cur_checkpoint = ModelCheckpoint('current.h5')
	val_checkpoint = ValCheckPoint(gen=val_gen, test_data=[x_dev, y_dev])

	def scheduler(epoch):
		if epoch < 5: return 1e-3
		if epoch < 8: return 1e-3 / 2
		return 1e-3 / 5

	reduce_lr = LearningRateScheduler(scheduler)

	try:
		model.fit_generator(train_gen, steps_per_epoch=num_train // batch_size[2], epochs=epochs, callbacks=[val_checkpoint, cur_checkpoint, reduce_lr])
	except KeyboardInterrupt:
		model.save_weights('interrupt.h5')

def get_score(preload):
	data = get_data()
	embedding_matrix = data['embedding_matrix']
	len_word_index = data['len_word_index']

	x_dev, y_dev = data['x_dev'], data['y_pos_dev']
	
	model = init_model(preload=preload, data=[embedding_matrix, len_word_index])
	test_gen = DataGen(x_dev, y_dev, test=True)

	get_metrics(model, test_gen, steps=269)

def main(args):
	mode, preload = args

	if preload == 'none': preload = None
	
	if mode == 'vis':
		data = get_data()
		model = init_model(data=[data['embedding_matrix'], data['len_word_index']])
		plot_model(model, to_file='fattie.png', show_shapes=True)

	elif mode == 'datavis':
		data = get_data()
		train_gen = DataGen(data['x_train'], data['y_pos_train'])
		data_visualizer(train_gen)

	elif mode == 'train':
		return runner(preload, 50)

	elif mode == 'score':
		return get_score(preload)

	else: raise ValueError('Incorrect mode')

if __name__ == '__main__':
	main(argv[1:])
