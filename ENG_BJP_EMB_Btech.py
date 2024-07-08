#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#@title Check GPU
get_ipython().run_line_magic('tensorflow_version', '2.x')
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
else:
  print('Found GPU at: {}'.format(device_name))


# In[ ]:


#@title Version Info
print('tf version: ', tf.__version__)


# In[ ]:


#@title Time
get_ipython().system('pip install ipython-autotime')

get_ipython().run_line_magic('load_ext', 'autotime')


# In[ ]:


#@title Import Libraries
from random import randint
from numpy import array
from numpy import argmax
import keras.backend as K
from tensorflow.keras import models
from numpy import array_equal
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Flatten,Embedding
from tensorflow.keras import Input
from tensorflow.keras.layers import TimeDistributed,Lambda
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import pad_sequences

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,unicodedata, re, io, time, gc, warnings
warnings.filterwarnings("ignore")

from tabulate import tabulate

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]


# In[ ]:


def d1():
  c=10
  d=10
  print(c)


# In[ ]:


d1()


# In[ ]:


#@title Function to Train & Test  given model (Early Stopping monitor 'val_loss')
def train_test(model, X_train, y_train , X_test, 	y_test,input_dict, output_dict, epochs=500, batch_size=32, patience=5,verbose=0):
	predicted_lst=[]
	actual_lst=[]
	# patient early stopping
	#es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1, patience=20)
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
	# train model
	print('training for ',epochs,' epochs begins with EarlyStopping(monitor= val_loss, patience=',patience,')....')
	history=model.fit(X_train, y_train, validation_split= 0.1, epochs=epochs,batch_size=batch_size, verbose=verbose, callbacks=[es])
	print(epochs,' epoch training finished...')

	# report training
	# list all data in history
	#print(history.history.keys())
	# evaluate the model
	_, train_acc = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
	_, test_acc = model.evaluate(X_test, 	y_test, batch_size=batch_size, verbose=0)
	print('\nPREDICTION ACCURACY (%):')
	print('Train: %.3f, Test: %.3f' % (train_acc*100, test_acc*100))
	# summarize history for accuracy
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title(model.name+' accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title(model.name+' loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()

	# spot check some examples
	space = 3*len(one_hot_decode(y_test[0]))
	print('10 examples from test data...')
	print('Input',' '*(space-4) ,'Expected',' '*(space-7) ,
	      'Predicted',' '*(space-5) ,'T/F')
	correct =0
	sampleNo =  10

	predicted= model.predict(X_test, batch_size=batch_size)

	for sample in range(0,sampleNo):
		if (one_hot_decode(y_test[sample])== one_hot_decode(predicted[sample])):
			correct+=1
		print(X_test[0][sample], ' ',
					one_hot_decode(y_test[sample]),' ', one_hot_decode(predicted[sample]),
					' ',one_hot_decode(y_test[sample])== one_hot_decode(predicted[sample]))
		predicted_lst.append(one_hot_decode(predicted[sample]))
		actual_lst.append(one_hot_decode(y_test[sample]))


	print('Accuracy: ', correct/sampleNo)
	for row in range(10):
		value_english = [i for val in X_test[0][row] for i in input_dict if input_dict[i]==val]
		print("English sentence is",' '.join(value_english).strip())
		actual_bjp = [i for val in actual_lst[row] for i in output_dict if output_dict[i]==val]
		print("Actual Bhojpuri is ",' '.join(actual_bjp).strip())
		predicted_bjp = [i for val in predicted_lst[row] for i in output_dict if output_dict[i]==val]
		print("Actual Bhojpuri is ",' '.join(predicted_bjp).strip())



# In[ ]:


#@title Data for Implementation
url = '/content/drive/MyDrive/Machine_Translation/Translation_MixDataset.xlsx'
df = pd.read_excel(url,header=None, names=['Eng','Bhoj','Trn'])

#Dropping the Trn Column
df.drop('Trn',axis=1, inplace=True)

#Shape
print("Total Records: ", df.shape[0])


# In[ ]:


#PreProcessing Function
def preprocess(w):
    w = w.lower().strip()  #lower case & remove white space
    w = re.sub(r"([?.!,¿।])", r" \1 ", w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w

#Applying PreProcess Function to a single sentence
x = np.random.randint(1,df.shape[0])
print("English: ", preprocess(df.Eng[x]))
print("Bhojpuri: ", preprocess(df.Bhoj[x]))

#applying the preprocess function
df['Eng'] = df['Eng'].apply(lambda x: preprocess(x))
df['Bhoj'] = df['Bhoj'].apply(lambda x: preprocess(x))

#Inspect
df.head()


# In[ ]:


#Tokenize Function
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')

    return tensor, lang_tokenizer


'''Tokenize Column Data'''
# Input = English || # Output = BJP
input_tensor, inp_lang = tokenize(df.loc[0:1000,'Eng'])
target_tensor, targ_lang = tokenize(df.loc[0:1000,'Bhoj'])


# In[ ]:


values=[160,2]


# In[ ]:


value = [i for val in values for i in inp_lang.word_index if inp_lang.word_index[i]==val]


# In[ ]:


value


# In[ ]:


inp_lang.word_index['spread']


# In[ ]:


input_tensor


# In[ ]:


inp_lang


# In[ ]:


max_length_eng = input_tensor.shape[1]
print("Maximum sentence length of English:", max_length_eng)
max_length_bjp = target_tensor.shape[1]
print("Maximum sentence length of Bhojpuri:", max_length_bjp)


# In[ ]:


# To train faster, we can limit the size of the dataset to 100 sentences
num_examples = 100

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 90-10 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.1)

# Show length
tb_data = [["Eng Train Tensor", len(input_tensor_train)], ["Bhoj Train Tensor",len(target_tensor_train)],
           ["Eng Val Tensor", len(input_tensor_val)], ["Bhoj Val Tensor",len(target_tensor_val)]]

print(tabulate(tb_data, headers=['','Lengths']))


# In[ ]:


input_tensor_train


# In[ ]:


def define_models(max_length_eng,max_length_bjp,vocab_size_eng,vocab_size_bjp):

  # Encoding layer english as input
  enc_input_lyr = Input(shape=(max_length_eng,))
  embedding_lyr = Embedding(vocab_size_eng,128)(enc_input_lyr)
  encoding_layer = LSTM(128, return_state=True)
  output_h, state_h, state_c = encoding_layer(embedding_lyr)
  states = [state_h, state_c]

  # decoding state
  dec_input_lyr = Input(shape=(max_length_bjp,), name='dec_input_lyr')
  decoder_embedding = Embedding(vocab_size_bjp,128)(dec_input_lyr)
  decoding_lstm = LSTM(128,return_sequences=True, return_state=True)
  decoder_outputs,_,_ = decoding_lstm(decoder_embedding,initial_state=states)
  outputs = TimeDistributed(Dense(vocab_size_bjp,activation='softmax'))(decoder_outputs)
  model_mt = Model([enc_input_lyr,dec_input_lyr],outputs)
  model_mt.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

  # define inference enoder
  encoder_model = Model(enc_input_lyr,states)
  # define decoder inference model
  decoder_state_input_h = Input(shape=(128,))
  decoder_state_input_c = Input(shape=(128,))
  decoder_states_inputs = [decoder_state_input_h,decoder_state_input_c]

  decoder_outputs, state_h, state_c = decoding_lstm(decoder_embedding,initial_state=decoder_states_inputs)
  decoder_states = [state_h,state_c]
  decoder_output_pred = Dense(vocab_size_bjp,activation='softmax')(decoder_outputs)
  decoder_model_pred = Model([dec_input_lyr]+decoder_states_inputs,[decoder_output_pred]+decoder_states)
  return model_mt,encoder_model,decoder_model_pred



# In[ ]:


# vocabulary sizes
vocab_size_eng = len(inp_lang.word_index)+1
vocab_size_bjp = len(targ_lang.word_index)+1
print("English vocabulary size is {0} and Bhojpri vocab size = {1}".format(vocab_size_eng,vocab_size_bjp))


# In[ ]:


model,enc_model,decoder_model = define_models(max_length_eng,max_length_bjp,vocab_size_eng,vocab_size_bjp)


# In[ ]:


plot_model(model, show_shapes=True)


# In[ ]:


model.summary()


# In[ ]:


enc_model.summary()


# In[ ]:


decoder_model.summary()


# In[ ]:


tar_encoded_train = to_categorical(target_tensor_train,num_classes=vocab_size_bjp)


# In[ ]:


tar_encoded_val = to_categorical(target_tensor_val,num_classes=vocab_size_bjp)


# In[ ]:


tar_encoded_train.shape


# In[ ]:


[input_tensor_train,target_tensor_train][0]


# In[ ]:


model.fit([input_tensor_train,target_tensor_train],tar_encoded_train,epochs=10)


# In[ ]:


train_test(model, [input_tensor_train,target_tensor_train],tar_encoded_train, [input_tensor_train,target_tensor_train],tar_encoded_train,inp_lang.word_index, targ_lang.word_index,epochs=30, batch_size=1, patience=5,verbose=2)


# ================END of the Project++++++++++++++++++
