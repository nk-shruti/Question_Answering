# from __future__ import absolute_import
from __future__ import division

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, InputLayer, TimeDistributed, Activation
from keras.layers import Reshape, Lambda, Highway
from keras.layers.core import Dense, RepeatVector, Masking, Dropout
from keras.layers.merge import Concatenate, Multiply, Add
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.initializers import Constant, Zeros
from keras.engine import Layer, InputSpec
import tensorflow as tf

def softmax(x, mask, axis=-1):
	"""Softmax activation function.
	# Arguments
		x : Tensor.
		axis: Integer, axis along which the softmax normalization is applied.
	# Returns
		Tensor, output of softmax transformation.
	# Raises
		ValueError: In case `dim(x) == 1`.
	"""
	e = mask * K.exp(mask * (x - K.max(x, axis=axis, keepdims=True)))
	s = K.sum(e, axis=axis, keepdims=True)
	return e / s

	# ndim = K.ndim(x)
	# if ndim == 2:
	# 	return K.softmax(x)
	# elif ndim > 2:
	# 	e = mask * K.exp(mask * (x - K.max(x, axis=axis, keepdims=True)))
	# 	s = K.sum(e, axis=axis, keepdims=True)
	# 	return e / s
	# else:
	# 	raise ValueError('Cannot apply softmax to a tensor that is 1D')

def BiDAF_mod(data, d=100, word2vec_dim=100, dropout_rate=0.2,
			char_level_embeddings=False, unroll=False):
	
	embedding_matrix, len_word_index = data

	X = Input(shape=(None,), name='context_input')
	Q = Input(shape=(None,), name='ques_input')

	embedding_layer = Embedding(len_word_index + 1,	word2vec_dim, trainable=False,
						embeddings_initializer=Constant(value=embedding_matrix), mask_zero=True)
	X_embedding = embedding_layer (X)
	Q_embedding = embedding_layer (Q)

	# X_embedding = TimeDistributed(Highway(activation='relu', input_dim=word2vec_dim), input_shape=(None, word2vec_dim)) (X_embedding)
	# X_embedding = TimeDistributed(Highway(activation='relu', input_dim=word2vec_dim), input_shape=(None, word2vec_dim)) (X_embedding)

	# Q_embedding = TimeDistributed(Highway(activation='relu', input_dim=word2vec_dim), input_shape=(None, word2vec_dim)) (Q_embedding)
	# Q_embedding = TimeDistributed(Highway(activation='relu', input_dim=word2vec_dim), input_shape=(None, word2vec_dim)) (Q_embedding)

	# Passage encoder
	H = Masking() (X_embedding)

	H = Masking() (X_embedding)
	# H = X_embedding
	for i in range(1):
		H = Bidirectional(GRU(units=d, return_sequences=True, dropout=dropout_rate,
								unroll=unroll)) (H)
	
	# Question encoder
	U = Masking() (Q_embedding)
	# U = Q_embedding
	for i in range(1):
		U = Bidirectional(GRU(units=d, return_sequences=True, dropout=dropout_rate,
								unroll=unroll)) (U)

	G = AttentionFlowLayer(d=d) ([H, U])

	# Modelling layer
	M = G
	for i in range(2):
		M = Bidirectional(GRU(units=d, return_sequences=True, dropout=dropout_rate,
								unroll=unroll)) (M)

	answer_start_input = Concatenate() ([G, M])
	answer_start_input = Dropout(dropout_rate) (answer_start_input)
	answer_start_weights = TimeDistributed(Dense(1)) (answer_start_input)
	answer_start = OutputLayer(name='as') (answer_start_weights)

	answer_end_input = AnswerEndInputCalc() ([G, M, answer_start])

	M2 = answer_end_input
	for i in range(1):
		M2 = Bidirectional(GRU(units=d, return_sequences=True, dropout=dropout_rate,
								unroll=unroll)) (M2)

	answer_end_input = Concatenate() ([G, M2])
	answer_end_input = Dropout(dropout_rate) (answer_end_input)
	answer_end_weights = TimeDistributed(Dense(1)) (answer_end_input)

	answer_end = OutputLayer(name='ae') (answer_end_weights)

	model = Model(inputs=[X, Q], outputs=[answer_start, answer_end])

	return model 

class AttentionFlowLayer(Layer):
	
	def __init__(self, d, **kwargs):
		self.d = d
		super(AttentionFlowLayer, self).__init__(**kwargs)
		self.supports_masking = True

	def build(self, input_shape=None):
		self.w = self.add_weight(name='w', shape=(1, 6 * self.d, 1, 1),
								 initializer='glorot_uniform', trainable=True)

		self.bias = self.add_weight(name='bias', shape=(1, 1, 1), initializer=Zeros(), trainable=True)


		super(AttentionFlowLayer, self).build(input_shape)


	def compute_mask(self, inputs, mask):
		return mask[0]

	def call(self, inputs, mask=None):
		H, U = inputs
		passage_mask = K.cast(K.expand_dims(mask[0], axis=1), 'float32')
		ques_mask = K.cast(K.expand_dims(mask[1], axis=1), 'float32')

		H = K.permute_dimensions(H, [0, 2, 1])	# 2d x T
		U = K.permute_dimensions(U, [0, 2, 1])	# 2d x J

		H = Multiply()([H, passage_mask])
		U = Multiply()([U, ques_mask])

		h_tiled = K.tile(K.expand_dims(H), K.concatenate([[1, 1, 1], [K.shape(U)[-1]]], 0))	# 2d x T x J
		u_tiled = K.tile(K.expand_dims(U, axis=2), K.concatenate([[1, 1],[ K.shape(H)[-1]], [1]], 0)) # 2d x T x J
		u_reshaped = K.expand_dims(U, axis=2) # 2d x 1 x J

		hu = Multiply() ([u_reshaped, h_tiled]) # 2d x T x J

		h_u_hu = K.concatenate([h_tiled, u_tiled, hu], axis=1) # 6d x T x J

		S = K.sum(Multiply() ([self.w, h_u_hu]), axis=1) + self.bias # T x J

		a = softmax(S, axis=-1, mask=ques_mask) # T x J
		U_ = K.batch_dot(U, a, axes=2) # 2d x T
		

		### Replace the masked values with something very -ve to not to be considered for max at all
		tiled_passage_mask = K.tile(K.permute_dimensions(passage_mask, [0, 2, 1]), K.concatenate([[1, 1], [K.shape(U)[-1]]], 0)) 
		tiled_ques_mask = K.tile(ques_mask, K.concatenate([[1, K.shape(H)[-1]], [1]], 0)) 

		similarity_matrix_max_mask = K.constant(-1e7) * Multiply() ([tiled_passage_mask, tiled_ques_mask])

		S_max_masked = Add() ([S, similarity_matrix_max_mask])
		S_max = K.max(S, axis=-1)
		S_max = Multiply() ([S_max, K.squeeze(passage_mask, axis=1)])
		b = softmax(S_max, mask=K.squeeze(passage_mask, axis=1))
		##############################################


		h_ = K.batch_dot(H, K.expand_dims(b, axis=-1), axes=[2, 1])
		H_ = K.tile(h_, K.concatenate([[1, 1], [K.shape(H)[-1]]], 0))
		
		G = K.concatenate([H, U_, Multiply()([H, U_]), Multiply()([H, H_])], axis=1)

		#compute mask:
		G = Multiply()([G, passage_mask])

		G = K.permute_dimensions(G, [0, 2, 1])

		return G

	def compute_output_shape(self, input_shape):
		
		return (input_shape[0][0], input_shape[0][1], 8*self.d)

class OutputLayer(Layer):

	def __init__(self, **kwargs):
		super(OutputLayer, self).__init__(**kwargs)
		self.supports_masking = True

	def build(self, input_shape=None):
		super(OutputLayer, self).build(input_shape)

	def compute_mask(self, inputs, mask):
		return None

	def call(self, inputs, mask=None):
		answer_weights = inputs
		mask = K.cast(mask, 'float32')
		answer_weights = K.squeeze(answer_weights, axis=-1)
		answer_probabs = softmax(answer_weights, mask=mask) # (, T)

		return answer_probabs

	def compute_output_shape(self, input_shape):
		
		return (input_shape[0], input_shape[1])

class AnswerEndInputCalc(Layer):

	def __init__(self, **kwargs):
		super(AnswerEndInputCalc, self).__init__(**kwargs)
		self.supports_masking = True

	def build(self, input_shape=None):
		super(AnswerEndInputCalc, self).build(input_shape)

	def compute_mask(self, inputs, mask):
		return mask[0]

	def call(self, inputs, mask=None):
		G, M, answer_start_probabs = inputs
		_, mask, _ = mask
		mask = K.cast(mask, 'float32')

		# shape of M: (, T, 2d)
		M_weighted = Multiply() ([M, K.expand_dims(answer_start_probabs, axis=2)])

		M_weighted_ans_start = K.tile(K.sum(M_weighted, axis=1, keepdims=True), 
									K.concatenate([[1, K.shape(answer_start_probabs)[1]], [1]], 0))

		answer_end_input = Concatenate() ([G, M, M_weighted_ans_start,
										 Multiply()([M, M_weighted_ans_start])])

		answer_end_input = Multiply()([answer_end_input, K.expand_dims(mask, axis=2)])

		return answer_end_input

	def compute_output_shape(self, input_shape):
		
		return (input_shape[1][0], input_shape[1][1], 7 * input_shape[1][2])

# class OutputLayer(Layer):

# 	def __init__(self, d, **kwargs):
# 		self.d = d
# 		super(OutputLayer, self).__init__(**kwargs)
# 		self.supports_masking = True

# 	def build(self, input_shape=None):
# 		self.w = self.add_weight(name='w', shape=(1, 10 * self.d, 1),
# 								 initializer='glorot_uniform', trainable=True)

# 		super(OutputLayer, self).build(input_shape)

# 	def compute_mask(self, inputs, mask):
# 		return None

# 	def call(self, inputs, mask=None):
# 		G, M = inputs
# 		mask = K.cast(K.expand_dims(mask[0], axis=1), 'float32')

# 		G = K.permute_dimensions(G, [0, 2, 1])
# 		M = K.permute_dimensions(M, [0, 2, 1])

# 		GM = K.concatenate([G, M], axis=1)

# 		GM = Multiply()([GM, mask])

# 		p = softmax(K.sum(Multiply() ([self.w, GM]), axis=1), mask=K.squeeze(mask, axis=1))

# 		return p

# 	def compute_output_shape(self, input_shape):
		
# 		return (input_shape[0][0], input_shape[0][1])

# if char_level_embeddings:
# 						P_str = Input(shape=(N, C), dtype='int32', name='P_str')
# 						Q_str = Input(shape=(M, C), dtype='int32', name='Q_str')
# 						input_placeholders = [P_vecs, P_str, Q_vecs, Q_str]

# 						char_embedding_layer = TimeDistributed(Sequential([
# 								InputLayer(input_shape=(C,), dtype='int32'),
# 								Embedding(input_dim=127, output_dim=H, mask_zero=True),
# 								Bidirectional(GRU(units=H))
# 						]))

# 						# char_embedding_layer.build(input_shape=(None, None, C))

# 						P_char_embeddings = char_embedding_layer(P_str)
# 						Q_char_embeddings = char_embedding_layer(Q_str)

# 						P = Concatenate() ([P_vecs, P_char_embeddings])
# 						Q = Concatenate() ([Q_vecs, Q_char_embeddings])

# def call(self, inputs):
# 		H, U = inputs

# 		# H = T x 2d
# 		# U = J x 2d

# 		## multiply each row vector of H to each row of U 

# 		h_tiled = K.tile(K.expand_dims(H, axis=2), K.concatenate([[1, 1], [K.shape(U)[1]], [1]], 0))
# 		u_tiled = K.tile(K.expand_dims(U, axis=1), K.concatenate([[1], [K.shape(H)[1]], [1, 1]], 0))

# 		hu = Multiply() ([h_tiled, u_tiled]) # T x J x 2d

# 		h_u_hu = K.concatenate([h_tiled, u_tiled, hu], axis=-1)
		
# 		h_u_hu = K.permute_dimensions(h_u_hu, [0, 3, 1, 2]) # batch_size x 2d x T x J
# 		H = K.permute_dimensions(H, [0, 2, 1])	# 2d x T
# 		U = K.permute_dimensions(U, [0, 2, 1])	# 2d x J

# 		S = K.sum(Multiply() ([self.w, h_u_hu]), axis=1)

# 		a = softmax(S, axis=-1)
# 		U_ = K.batch_dot(U, a, axes=2)

# 		b = softmax(K.max(S, axis=-1))

# 		h_ = K.batch_dot(H, K.expand_dims(b, axis=-1), axes=[2, 1])
# 		H_ = K.tile(h_, K.concatenate([[1, 1], [K.shape(H)[-1]]], 0))
		
# 		G = K.concatenate([H, U_, Multiply()([H, U_]), Multiply()([H, H_])], axis=1)

# 		return K.permute_dimensions(G, [0, 2, 1])