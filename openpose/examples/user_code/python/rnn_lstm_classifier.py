'''
File Description: RNN LSTM MultiClass Classifier for Openpose Keypoints
File Name       : rnn_lstm_classifier.py
Author          : Srini Ananthakrishnan
Github          : https://github.com/srianant/
Date            : 09/23/2017
'''

# Import packages
import IPython
import pandas as pd
import keras
import itertools
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, TimeDistributed, Activation
from keras.layers import LSTM
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from kerasify import export_model
from keras.models import model_from_json

def samples_to_3D_array(_vector_dim, _vectors_per_sample, _X):
    '''
    Keras LSTM model require 3-Dimensional Tensors.
    Function convert samples to 3D tensors !
    '''
    X_len = len(_X)
    result_array = []
    for sample in range (0,X_len):
        sample_array = []
        for vector_idx in range (0, _vectors_per_sample):
            start = vector_idx * _vector_dim
            end = start + _vector_dim
            sample_array.append(_X[sample][start:end])
        result_array.append(sample_array)

    return np.asarray(result_array)

def convert_y_to_one_hot(_y):
    '''
    Converst y integer labels (0,1,2..) to one_hot_encoding vectors
    '''
    _y = np.asarray(_y,dtype=int)
    b = np.zeros((_y.size, _y.max()+1))
    b[np.arange(_y.size),_y] = 1
    return b


def main():
    '''
    main routine to train and generate keras classifier model
    '''
    # Model Parameters and Paths
    # NOTE: Uncomment below cell to generate one of the classifier model (hand or pose or face)

    timesteps = 5

    ## POSE - val_acc: 98%
    epochs = 1000
    batch_size = 32
    _dropout = 0.1
    _activation='relu'
    _optimizer='Adam'
    class_names = ["close_to_camera","standing","sitting"]
    X_vector_dim = 36 # number of features or columns (pose)
    samples_path = "../../../train_data/pose/pose_samples_raw.txt"
    labels_path = "../../../train_data/pose/pose_labels_raw.txt"
    model_path = '../../../train_data/pose/pose.model'
    json_model_path = '../../../train_data/pose/pose_model.json'
    model_weights_path = "../../../train_data/pose/pose_model.h5"

    # ## HAND - val_acc: 97%
    # epochs = 1000
    # batch_size = 32
    # _dropout = 0.1
    # _activation='relu'
    # _optimizer='adam'
    # class_names = ["fist","pinch","wave","victory","stop","thumbsup"]
    # X_vector_dim = 40 # number of features or columns (hand)
    # samples_path = "../../../train_data/hand/hand_samples_raw.txt"
    # labels_path = "../../../train_data/hand/hand_labels_raw.txt"
    # model_path = '../../../train_data/hand/hand.model'
    # json_model_path = '../../../train_data/hand/hand_model.json'
    # model_weights_path = "../../../train_data/hand/hand_model.h5"

    # # ## FACE - val_acc: 97%
    # epochs = 1000
    # batch_size = 32
    # _dropout = 0.1
    # _activation='tanh'
    # _optimizer='Adadelta'
    # class_names = ["normal","happy","sad","surprise"]
    # X_vector_dim = 96 # number of features or columns (face)
    # samples_path = "../../../train_data/face/face_samples_raw.txt"
    # labels_path = "../../../train_data/face/face_labels_raw.txt"
    # model_path = '../../../train_data/face/face.model'
    # json_model_path = '../../../train_data/face/face_model.json'
    # model_weights_path = "../../../train_data/face/face_model.h5"

    # Load Keypoints Samples and Labels
    X = np.loadtxt(samples_path, dtype="float")
    y = np.loadtxt(labels_path)

    y_one_hot = convert_y_to_one_hot(y) # convert to one_hot_encoding vector
    y_vector_dim = y_one_hot.shape[1] # number of features or columns

    X_vectors_per_sample = timesteps # number of vectors per sample

    # Keras LSTM model require 3-Dimensional Tensors. Convert samples to 3D tensors
    X_3D = samples_to_3D_array(X_vector_dim, X_vectors_per_sample, X)

    # Perform test-train split
    X_train, X_test, y_train, y_test = train_test_split(X_3D, y_one_hot, test_size=0.33, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2]) # store input_shape

    print "Model Parameters:"
    print "input_shape     : ", input_shape
    print "X_vector_dim    : ", X_vector_dim
    print "y_vector_dim    : ", y_vector_dim

    # Build Keras TimeDistributed(Dense) (many-to-many case) LSTM model
    print ("Build Keras Timedistributed-LSTM Model...")
    model = Sequential()
    model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim*2, activation=_activation))) #(5, 80)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation))) #(5, 40)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim/2, activation=_activation))) #(5, 20)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim/4, activation=_activation))) #(5, 10)
    model.add(Dropout(_dropout))
    model.add(LSTM(X_vector_dim/4, dropout=_dropout, recurrent_dropout=_dropout))
    model.add(Dense(y_vector_dim,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=_optimizer, metrics=['accuracy'])
    model.summary()

    # Fit model
    print('Training...')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test))

    # Evaluate Model and Predict Classes
    print('Testing...')
    score, accuracy = model.evaluate(X_test, y_test,
                                     batch_size=batch_size)

    print('Test score: {:.3}'.format(score))
    print('Test accuracy: {:.3}'.format(accuracy))

    # Export model
    export_model(model,model_path)

    # serialize model to JSON
    model_json = model.to_json()
    with open(json_model_path, "w") as json_file:
        json_file.write(json_model_path)
    # serialize weights to HDF5
    model.save_weights(model_weights_path)
    print("Saved model to disk")


if __name__ == "__main__":
    main()
