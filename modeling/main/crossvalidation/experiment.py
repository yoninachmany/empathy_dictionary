import pandas as pd
import os
import util
import modeling.feature_extraction as fe
from modeling import common
from scipy import stats as st
from sklearn import metrics
import numpy as np
import keras
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold
# To get word-level ratings, will need to tokenize texts.
from nltk.tokenize import wordpunct_tokenize as tokenize
from nltk.util import ngrams

## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k
 
###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################


#######		Setting up data		########
train, dev, test=util.train_dev_test_split(util.get_messages())
data=pd.concat([train,test], axis=0) #excluding dev set from CV
data=data.reset_index(drop=True)
########################################

# print(data, data.shape)
# assert(False)

results_df=pd.DataFrame(
	index=['cnn', 'ffn', 'ridge', 'maxent'], 
	columns=['empathy', 'empathy_bin', 'distress', 'distress_bin'])


embs=common.get_facebook_fasttext_common_crawl(vocab_limit=None)

# Based on column names in df, determine if empathy/vad.
dataset = 'vad_eb' if 'text' in data else 'empathy'
TARGETS=['empathy', 'distress']


if dataset == 'vad_eb':
	TARGETS=['V', 'A', 'D']
	# data.essay used later
	data['essay']=data.text




# features_train_centroid=fe.embedding_centroid(train.essay, embs)
# features_train_matrix=fe.embedding_matrix(train.essay, embs, common.TIMESTEPS)

# features_test_centroid=fe.embedding_centroid(test.essay, embs)
# features_test_matrix=fe.embedding_matrix(test.essay, embs, common.TIMESTEPS)

FEATURES_MATRIX=fe.embedding_matrix(data.essay, embs, common.TIMESTEPS)
FEATURES_CENTROID=fe.embedding_centroid(data.essay, embs)

# Get (lowercase) tokens and their features, for predicting word ratings from embeddings.
TOKENS_ESSAYS=[tokenize(essay.lower()) for essay in data.essay]
print(len(TOKENS_ESSAYS))
unigrams = set([ngrams(tokens, 1) for tokens in TOKENS_ESSAYS])
print(len(unigrams))
bigrams = set([ngrams(tokens, 2) for tokens in TOKENS_ESSAYS])
print(len(bigrams))
trigrams = set([ngrams(tokens, 3) for tokens in TOKENS_ESSAYS])
print(len(trigrams))
TOKENS = list(unigrams.union(bigrams).union(trigrams))
print(len(TOKENS))
TOKENS_CENTROID=fe.embedding_centroid(TOKENS, embs)

# Save tokens, token centroids, and feature centroids for SHAP.
# See https://github.com/slundberg/shap/blob/master/notebooks/deep_explainer/Keras%20LSTM%20for%20IMDB%20Sentiment%20Classification.ipynb.
np.save('results/{}_tokens'.format(dataset) , TOKENS)
np.save('results/{}_tokens_centroid'.format(dataset), TOKENS_CENTROID)
np.save('results/{}_features_centroid'.format(dataset), FEATURES_CENTROID)
# LABELS={
# 	'empathy':{'classification':'empathy_bin', 'regression':'empathy'},
# 	'distress':{'classification':'distress_bin', 'regression':'distress'}
# }

def f1_score(true, pred):
	pred=np.where(pred.flatten() >.5 ,1,0)
	result=metrics.precision_recall_fscore_support(
		y_true=true, y_pred=pred, average='micro')
	return result[2]

def correlation(true, pred):
	pred=pred.flatten()
	result=st.pearsonr(true,pred)
	return result[0]

# SCORE={
# 	'classification': f1_score,
# 	'regression':correlation
# }


MODELS={
	'cnn':lambda:common.get_cnn(
							input_shape=[common.TIMESTEPS,300], 
							num_outputs=1, 
							num_filters=100, 
							learning_rate=1e-3,
							dropout_conv=.5, 
							problem='regression'),

	'ffn': lambda:	common.get_ffn(
							units=[300,256, 128,1], 
							dropout_hidden=.5,
							dropout_embedding=.2, 
							learning_rate=1e-3,
							problem='regression'),

	'ridge': lambda: RidgeCV(
							alphas=[1, 5e-1, 1e-1,5e-2, 1e-2, 5e-3, 1e-3,5e-4, 1e-4])
}



early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=20,
                              verbose=0, mode='auto')


num_splits=10

performancens={name:pd.DataFrame(columns=['empathy', 'distress'], 
	index=range(1,num_splits+1)) for name in MODELS.keys()}


"""
# Save the fold for which the text appears in the test set, for other methods.
id_to_test_fold = {"id": [], "test_fold": []}

kf_iterator=KFold(n_splits=num_splits, shuffle=True, random_state=42)
for i, splits in enumerate(kf_iterator.split(data)):
	train,test=splits

	# For all texts in the test set, add mapping from id to the current fold.
	for test_id in test:
		id_key = "id" if dataset == "vad" else "message_id"
		id_to_test_fold["id"].append(data[id_key][test_id])
		id_to_test_fold["test_fold"].append(i)

	k.clear_session()

	for target in TARGETS:
		print(target)

		labels_train=data[target][train]
		labels_test=data[target][test]

		features_train_centroid=FEATURES_CENTROID[train]
		features_train_matrix=FEATURES_MATRIX[train]

		features_test_centroid=FEATURES_CENTROID[test]
		features_test_matrix=FEATURES_MATRIX[test]


		# print(labels_train)
		# print(features_train_matrix)

		# Hack to not de-indent body of for loop when only using FFN model.
		for j in range(1):
			model_name = 'ffn'
			model_fun = MODELS[model_name]
			model=model_fun()


			#	TRAINING
			if model_name=='cnn':
				model.fit(	features_train_matrix, 
							labels_train,
							epochs=200, 
							batch_size=32, 
							validation_split=.1, 
							callbacks=[early_stopping])

			elif model_name=='ffn':
				model.fit(	features_train_centroid,
							labels_train,
							epochs=200, 
							validation_split=.1, 
							batch_size=32, 
							callbacks=[early_stopping])

			elif model_name=='ridge':
				model.fit(	features_train_centroid,
							labels_train)

			else:
				raise ValueError('Unkown model name encountered.')

			# Get tokens in test texts to predict on, with model trained on train texts.
			tokens_test = list(set([token.lower() for essay in data.essay[test] for token in tokenize(essay)]))
			tokens_test_centroid = fe.embedding_centroid(tokens_test, embs)

			# Predict ratings for test tokens.
			pred = model.predict(tokens_test_centroid)[:, 0]

			# Save ratings as DataFrame.
			ratings = {'tokens': tokens_test, 'ratings': pred}
			ratings_df = pd.DataFrame.from_dict(ratings)
			ratings_df = ratings_df[['tokens', 'ratings']]
			ratings_df.to_csv('results/{}_ratings_{}.tsv'.format(target, i),
							  sep='\t')


			#	PREDICTION
			# if model_name=='cnn':
			# 	pred=model.predict(features_test_matrix)
			# else:
			# 	pred=model.predict(features_test_centroid)

			#	SCORING
			# result=correlation(true=labels_test, pred=pred)

			#	RECORD
			# row=model_name
			# column=LABELS[target][problem]
			# results_df.loc[row,column]=result
			# print(results_df)
			# performancens[model_name].loc[i+1,target]=result
			# print(performancens[model_name])

# Write the mapping from text id to fold for which it appears in test set, for other methods.
id_to_test_fold_df = pd.DataFrame.from_dict(id_to_test_fold)
id_to_test_fold_df.to_csv('results/{}_id_to_test_fold.tsv'.format(dataset), sep='\t')
"""
# Comment back in to get the full lexica.


# Set random seed for reproducibility, like random_state in the KFold.
np.random.seed(42)

for target in TARGETS:
	# clear_session() was done outside loop when there was an outer loop.
	k.clear_session()
	print(target)

	# Focus only on FFN, which easily maps between features and words.
	model=MODELS['ffn']()

	# Train on all features.
	model.fit(FEATURES_CENTROID,
			data[target],
			epochs=200,
			validation_split=.1,
			batch_size=32,
			callbacks=[early_stopping])

	# Save model for SHAP.
	# model.save('results/model_{}.h5'.format(target))

	# Predict ratings for all tokens.
	pred=model.predict(TOKENS_CENTROID)[:, 0]

	# Save ratings as DataFrame.
	ratings={'tokens': TOKENS, 'ratings': pred}
	ratings_df=pd.DataFrame.from_dict(ratings)
	ratings_df=ratings_df[['tokens', 'ratings']]
	ratings_df.to_csv('results/{}_ratings.tsv'.format(target), sep='\t')


"""
#average results data frame
if not os.path.isdir('results'):
	os.makedirs('results')
for key, performance in performancens.items():
	performance['mean']=performance.mean(axis=1)
	mean=performance.mean(axis=0)
	stdev=performance.std(axis=0)
	performance.loc['stdev']=stdev
	performance.loc['mean']=mean
	performance.to_csv('results/{}.tsv'.format(key), sep='\t')

# results_df.to_csv('results.tsv', sep='\t')

"""




