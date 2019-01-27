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
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
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
noc, noq = 128, 1
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
		model.load_weights(preload, by_name=True)

	model.summary()

	print 'Done!\n'
	return model

def onehot(x, size):
	X = np.zeros((len(x), size), dtype=np.float32)
	for i, xx in enumerate(x):
		X[i][xx] = 1.
	return X

def indices_generator(num_samples, sequentially):
	if not sequentially:
		while True:
			indices = np.random.choice(num_samples, size=min(noc, num_samples), replace=False)
			yield indices
	else:
		idx = np.arange(num_samples)
		batch_indices = [idx[i:min(i + noc, num_samples)] for i in xrange(0, num_samples, noc)]

		for batch in batch_indices: yield batch

def pad_char_embeddings(embeddings, size):
	embeddings = embeddings.tolist()
	for i in xrange(len(embeddings)):
		if len(embeddings[i]) < size:
			embeddings[i] += [[0] * MAX_WORD_LEN] * (size - len(embeddings[i]))
	return np.asarray(embeddings, dtype=np.float32)

def DataGen(X, y, sequentially=False):
	(contexts, char_contexts), (questions, char_questions) = X
	ans_starts, ans_ends = y

	# contexts, questions = contexts[:500], questions[:500]
	# ans_starts, ans_ends = ans_starts[:500], ans_ends[:500]

	c_bins = [[] for _ in xrange(7)]
	cc_bins = [[] for _ in xrange(7)]
	q_bins = [[] for _ in xrange(7)]
	cq_bins = [[] for _ in xrange(7)]
	as_bins = [[] for _ in xrange(7)]
	ae_bins = [[] for _ in xrange(7)]

	def put(c, cc, q, cq, anss, anse, at):
		c_bins[at].append(c)
		cc_bins[at].append(cc)
		q_bins[at].append(q)
		cq_bins[at].append(cq)
		as_bins[at].append(anss)
		ae_bins[at].append(anse)

	index_gen = indices_generator(len(contexts), sequentially=sequentially)

	while True:
		
		try: indices = index_gen.next()
		except StopIteration:
			for i in xrange(len(c_bins)):
				if len(c_bins[i]) > 0:
					cur_c = c_bins[i]
					cur_char_c = cc_bins[i]
					cur_q = q_bins[i]
					cur_char_q = cq_bins[i]
					cur_as = as_bins[i]
					cur_ae = ae_bins[i]

					max_context_len_of_batch = len(max(cur_c, key=len))
					max_q_len_of_batch = len(max(cur_q, key=len))

					cur_c = pad_sequences(cur_c, maxlen=max_context_len_of_batch, padding='post')
					cur_q = pad_sequences(cur_q, maxlen=max_q_len_of_batch, padding='post')

					cur_char_c = pad_char_embeddings(np.asarray(cur_char_c), size=max_context_len_of_batch)
					cur_char_q = pad_char_embeddings(np.asarray(cur_char_q), size=max_q_len_of_batch)

					cur_as = onehot(cur_as, size=cur_c.shape[1])
					cur_ae = onehot(cur_ae, size=cur_c.shape[1])

					yield [cur_c, cur_q, cur_char_c, cur_char_q], [cur_as, cur_ae]
			return

		cur_c, cur_char_c, cur_q, cur_char_q, cur_as, cur_ae = (contexts[indices], char_contexts[indices], questions[indices], 
										char_questions[indices], ans_starts[indices], ans_ends[indices])

		for c, cc, q, cq, anss, anse in zip(cur_c, cur_char_c, cur_q, cur_char_q, cur_as, cur_ae):
			put(c, cc, q, cq, anss, anse, at=min(6, len(c) // 50))

		for i in xrange(len(c_bins)):
			if len(c_bins[i]) >= batch_size[i]:
				idx = len(c_bins[i]) - batch_size[i]

				cur_c = c_bins[i][idx:]
				cur_char_c = cc_bins[i][idx:]
				cur_q = q_bins[i][idx:]
				cur_char_q = cq_bins[i][idx:]
				cur_as = as_bins[i][idx:]
				cur_ae = ae_bins[i][idx:]

				del c_bins[i][idx:], cc_bins[i][idx:], q_bins[i][idx:], cq_bins[i][idx:], as_bins[i][idx:], ae_bins[i][idx:]

				max_context_len_of_batch = len(max(cur_c, key=len))
				max_q_len_of_batch = len(max(cur_q, key=len))

				cur_c = pad_sequences(cur_c, maxlen=max_context_len_of_batch, padding='post')
				cur_q = pad_sequences(cur_q, maxlen=max_q_len_of_batch, padding='post')

				cur_char_c = pad_char_embeddings(np.asarray(cur_char_c), size=max_context_len_of_batch)
				cur_char_q = pad_char_embeddings(np.asarray(cur_char_q), size=max_q_len_of_batch)

				cur_as = onehot(cur_as, size=cur_c.shape[1])
				cur_ae = onehot(cur_ae, size=cur_c.shape[1])
				
				yield [cur_c, cur_q], [cur_as, cur_ae]

def runner(preload, epochs):
	data = get_data()
	embedding_matrix = data['embedding_matrix']
	len_word_index = data['len_word_index']
	
	x_train, y_train = data['x_train'], data['y_pos_train']
	x_dev, y_dev = data['x_dev'], data['y_pos_dev']
	
	model = init_model(preload=preload, data=[embedding_matrix, len_word_index])
	
	val_checkpoint = ModelCheckpoint('bestval.h5','val_loss', 1, True) 
	cur_checkpoint = ModelCheckpoint('current.h5')
	# save_ema = ExponentialMovingAverage() 

	print 'Model compiled.'
	train_gen, val_gen = DataGen(x_train, y_train), DataGen(x_dev, y_dev)
	try:
		model.fit_generator(train_gen, steps_per_epoch=num_train // batch_size[2], epochs=epochs, validation_data=val_gen, 
					validation_steps=num_val // batch_size[2], callbacks=[val_checkpoint, cur_checkpoint])
	except KeyboardInterrupt:
		model.save_weights('interrupt.h5')

def get_score(preload):
	data = get_data()
	embedding_matrix = data['embedding_matrix']
	len_word_index = data['len_word_index']

	x_dev, y_dev = data['x_dev'], data['y_pos_dev']
	
	model = init_model(preload=preload, data=[embedding_matrix, len_word_index])
	test_gen = DataGen(x_dev, y_dev, sequentially=True)

	get_metrics(model, test_gen, steps=222)

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


def old_DataGen(X, y, validation=False):
	contexts, questions = X
	ans_starts, ans_ends = y[0], y[1]

	contexts, questions = contexts[:20], questions[:20]
	ans_starts, ans_ends = ans_starts[:20], ans_ends[:20]

	c_bins = [[] for _ in xrange(7)]
	q_bins = [[] for _ in xrange(7)]
	as_bins = [[] for _ in xrange(7)]
	ae_bins = [[] for _ in xrange(7)]

	def put(c, q, anss, anse, at):
		c_bins[at].append(c)
		q_bins[at].append(q)
		as_bins[at].append(anss)
		ae_bins[at].append(anse)

	while True:
		cur_c = []
		cur_q = []
		cur_as = []
		cur_ae = []

		para_indices = np.random.choice(len(contexts), size=min(nop, len(contexts)), replace=False)

		context_paras = contexts[para_indices].tolist()
		ques_paras = questions[para_indices].tolist()
		ans_start_paras = ans_starts[para_indices].tolist()
		ans_end_paras = ans_ends[para_indices].tolist()

		for i, (context_para, ques_para, ans_start_para, ans_end_para) in enumerate(zip(context_paras, ques_paras, ans_start_paras, ans_end_paras)):
			context_indices = np.random.choice(len(context_para), size=min(noc, len(context_para)), replace=False)

			for ci in context_indices:
				cur_c.append(context_para[ci])

				##### choosing 1 q for each context !!!! ######
				qi = randint(0, len(ques_para[ci]) - 1)
				
				cur_q.append(ques_para[ci][qi])
				try:
					cur_as.append(ans_start_para[ci][qi][0])
				except IndexError:
					print i, ci, qi
				cur_ae.append(ans_end_para[ci][qi][0])

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

				try:
					cur_as = onehot(cur_as, size=cur_c.shape[1])
					cur_ae = onehot(cur_ae, size=cur_c.shape[1])
				except IndexError:
					print 'Continued'
					continue

				yield [cur_c, cur_q], [cur_as, cur_ae]