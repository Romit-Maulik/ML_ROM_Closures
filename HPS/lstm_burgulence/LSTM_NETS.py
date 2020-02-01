from tensorflow.keras.layers import Input, Dense, Lambda, Add, LSTM
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model, Sequential

import numpy as np


def lstm_for_dynamics(cf_trunc,deployment_mode='test'):
    # LSTM hyperparameters
    seq_num = 30
    num_units = 73
    lrate = 0.0005440360402
    rho  = 0.998848446841937
    decay = 6.1587540045897e-06
    num_epochs = 317
    batch_size = 23

    features = np.transpose(cf_trunc)
    states = np.copy(features[:,:]) #Rows are time, Columns are state values

    # Need to make batches of 10 input sequences and 1 output
    total_size = np.shape(features)[0]-seq_num
    input_seq = np.zeros(shape=(total_size,seq_num,np.shape(states)[1]))
    output_seq = np.zeros(shape=(total_size,np.shape(states)[1]))

    for t in range(total_size):
        input_seq[t,:,:] = states[None,t:t+seq_num,:]
        output_seq[t,:] = states[t+seq_num,:]

    idx = np.arange(total_size)
    np.random.shuffle(idx)
    
    input_seq = input_seq[idx,:,:]
    output_seq = output_seq[idx,:]
    
    # Model architecture
    model = Sequential()
    model.add(LSTM(num_units,input_shape=(seq_num, np.shape(states)[1])))  # returns a sequence of vectors of dimension 32
    model.add(Dense(np.shape(states)[1], activation='linear'))

    # design network
    my_adam = optimizers.RMSprop(lr=lrate, rho=rho, epsilon=None, decay=decay)

    filepath = "best_weights_lstm.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
    callbacks_list = [checkpoint]
    
    # fit network
    model.compile(optimizer=my_adam,loss='mean_squared_error',metrics=[coeff_determination])

    if deployment_mode == 'train':
        train_history = model.fit(input_seq, output_seq, epochs=num_epochs, batch_size=batch_size, validation_split=0.33, callbacks=callbacks_list)#validation_split = 0.1
        np.save('Train_Loss.npy',train_history.history['loss'])
        np.save('Val_Loss.npy',train_history.history['val_loss'])

    model.load_weights(filepath)

    

    return model

def evaluate_rom_deployment_lstm(model,dataset,tsteps):

    seq_num = 30

    # Make the initial condition from the first seq_num columns of the dataset
    features = np.transpose(dataset)  
    input_state = np.copy(features[0:seq_num,:])

    state_tracker = np.zeros(shape=(1,int(np.shape(tsteps)[0]),np.shape(features)[1]),dtype='double')
    state_tracker[0,0:seq_num,:] = input_state[0:seq_num,:]
    
    for t in range(seq_num,int(np.shape(tsteps)[0])):
        lstm_input = state_tracker[:,t-seq_num:t,:]
        output_state = model.predict(lstm_input)
        state_tracker[0,t,:] = output_state[:]

    return np.transpose(output_state), np.transpose(state_tracker[0,:,:])

def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )