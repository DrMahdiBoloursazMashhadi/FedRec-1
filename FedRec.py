# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:42:09 2021

@author: Mahdi
"""

import os
import collections
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow_federated as tff

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MC = 5 # Number of Monte Carlo trials
iid = False # set to False to simulate non-iid user fading 

### Modulation Parameters
Q = 4 
M = 2**Q # 16QAM modulation
Es = 10 # Average symbol energy

### Noise Parameters
EbN0_dB = 5 # SNR per bit (dB)
EsN0_dB = EbN0_dB + 10*np.log10(Q) # SNR per symbol (dB)
N0 = Es/10**(EsN0_dB/10)

### FedRec Training Parameters
U = 5 # Number of wireless users taking part in federated training
EPOCH = 5 # Number of local epochs for each aggragation round
AGGREGATION_ROUND = 5 # Number of federated aggregation rounds
BATCH_SIZE = 20 
SHUFFLE_BUFFER = 20
PREFETCH_BUFFER = 10

### Load QAM I/Q Symbols
TRAIN = sio.loadmat('TRAIN.mat')
Train = TRAIN['TRAIN']
TEST = sio.loadmat('TEST.mat')
Test = TEST['TEST']

### Generate Real-valued Train/Test Features/Labels
N_T = int(Train.shape[0]/U) # Size of local user datasets 
x_train = np.zeros((N_T,2))
x_train[:,0] = np.real(Train[0:N_T,0])
x_train[:,1] = np.imag(Train[0:N_T,0])
y_train = Train[0:N_T,1]-1

x_test = np.zeros((Test.shape[0],2))
x_test[:,0] = np.real(Test[:,0])
x_test[:,1] = np.imag(Test[:,0])
y_test = Test[:,1]-1


################################ Functions ################################

def usergen(x, y):
    if iid==True:
        sigma = 1 # i.i.d. user fading 
    else:
        sigma = np.random.uniform(low=0.5, high=1.5, size=None) #non-i.i.d. user fading
        
    h = np.random.rayleigh(scale=sigma, size=(x.shape[0],1))
    hIQ=np.concatenate((h,h),axis=1)
    x_u = np.multiply(hIQ,x)+ np.random.normal(0 , np.sqrt(N0)/2 , x.shape) # Channel-distorted noisy features
    y_u = y # Labels
    D_u = tf.data.Dataset.from_tensor_slices((list(x_u),list(y_u.astype(int))))
    return D_u


def testgen(x, y):
    if iid==True:
        h = np.random.rayleigh(scale=1, size=(x.shape[0],1)) # i.i.d. test fading 
    else:
        h = np.random.rayleigh(scale=np.random.uniform(low=0.5, high=1.5, size=(x.shape[0],1)), size=(x.shape[0],1)) # non-i.i.d. user fading
    
    hIQ = np.concatenate((h,h),axis=1)
    x_test = np.multiply(hIQ,x) + np.random.normal(0 , np.sqrt(N0)/2 , x.shape) # Channel-distorted noisy features
    y_test = y # Labels
    dataset_test = tf.data.Dataset.from_tensor_slices((list(x_test),list(y_test.astype(int))))
    return dataset_test 


def preprocess(dataset):
  def batch_format_fn(element1,element2):
    return collections.OrderedDict(
        x=tf.reshape(element1, [-1, 2]),
        y=tf.reshape(element2, [-1, 1]))
  return dataset.repeat(EPOCH).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(2,)),
      tf.keras.layers.Dense(M),
      tf.keras.layers.Softmax(),])


def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    
################################ Main ################################

example_dataset = usergen(x_train, y_train)
preprocessed_example_dataset=preprocess(example_dataset)
example_element = next(iter((preprocessed_example_dataset)))     


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.05),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))


evaluation = tff.learning.build_federated_evaluation(model_fn)


BER = 0
for i in range(MC):
    
    print('##############################')
    print('Monte Carlo Trial # ', i+1)
    
    ### Generate Federated User Datasets
    federated_train_data=[]
    for u in range(U):
        D_u = usergen(x_train, y_train) ### Generate Local Dataset at user u
        federated_train_data.append(preprocess(D_u))
            
    ### Generate Test Dataset
    test_dataset = testgen(x_test, y_test)
    federated_test_data=[preprocess(test_dataset)]
    
    ### Federated Training
    state = iterative_process.initialize()
    for n in range(AGGREGATION_ROUND):
      state, metrics = iterative_process.next(state, federated_train_data)
      print(str(metrics))
      
    ### Evaluate the Model
    test_metrics = evaluation(state.model, federated_test_data)
    print(str(test_metrics))
    
    BER = BER + (1-test_metrics['sparse_categorical_accuracy'])/(Q*MC)

    
print('##############################')
print('16QAM at Eb/N0=', EbN0_dB, 'dB')
print('FedRec trained collaboratively by ', U, 'users')
if iid==True:
        iidstr = 'iid' 
else:
        iidstr = 'non-iid'
print(iidstr, ' Rayleigh fading')
print('BER= ', BER)
