import argparse
import logging

import os
import keras
import numpy as np
from keras import backend as K
from keras.callbacks import TensorBoard

import nni

from keras.models import Sequential
from keras.layers import GRU, Dense

from pandas import read_csv
from sklearn.preprocessing import LabelBinarizer

LOG = logging.getLogger('mnist_soc_annot')

look_back = 15
feature = 3  # voltage and current and temp

logger = logging.getLogger('mnist_AutoML')

class socNetwork(object):
    '''
    For initializing and building the deep learning network
    '''
    
    
    def __init__(self, 
                 hidden_size_1, hidden_size_2, hidden_size_3):
        """@nni.variable(nni.choice(8, 64, 128), name=self.hidden_size_1)"""
        self.hidden_size_1 = hidden_size_1
        """@nni.variable(nni.choice(8, 64, 128), name=self.hidden_size_2)"""
        self.hidden_size_2 = hidden_size_2
        """@nni.variable(nni.choice(8, 64, 128), name=self.hidden_size_3)"""
        self.hidden_size_3 = hidden_size_3
        
        
    def build_network(self):
        '''
        Build the network
        '''
        model = Sequential()
        model.add(GRU(self.hidden_size_1, input_shape=(look_back, feature), return_sequences=True))
        model.add(GRU(self.hidden_size_2, return_sequences=True))
        model.add(GRU(self.hidden_size_3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
        return model
    
    def train(self, trainX, trainY, textX, testY):
        model = self.build_network()
        model.fit(x=trainX, y=trainY, validation_split=0.15, epochs=5)
        
        _, score = model.evaluate(textX, testY, verbose=0)
        logger.debug('Final result is: %d', score)
        logger.debug('Send final result done.')
        """@nni.report_final_result(score)"""

        

def main(params):
    '''
    Main function, build mnist network, run and send result to NNI.
    '''
    dataframe_trainset = read_csv(
    'dataset/different_cycle_train_test/train_FUDS_BJDST_US06.csv', header=0, index_col=0)
    dataframe_testset = read_csv(
    'dataset/different_cycle_train_test/test_DST.csv', header=0, index_col=0)
    
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            # use first and second column in the dataset
            a = dataset[i:(i+look_back), 0:3]
            dataX.append(a)
            # the output is at the third column of the dataset
            dataY.append(dataset[i + look_back, 3])
        return np.array(dataX), np.array(dataY)
    
    train_dataset = dataframe_trainset.values
    test_dataset = dataframe_testset.values
    
    trainX, trainY = create_dataset(train_dataset, look_back)
    testX, testY = create_dataset(test_dataset, look_back)
    
    soc_network = socNetwork(hidden_size_1=params['hidden_size_1'], hidden_size_2=params['hidden_size_2'], hidden_size_3=params['hidden_size_3'])
    soc_network.build_network()
    soc_network.train(trainX, trainY, testX, testY)
        


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=str, default='/tmp/tensorflow/mnist/input_data', help="data directory")
    # parser.add_argument("--dropout_rate", type=float, default=0.5, help="dropout rate")
    # parser.add_argument("--channel_1_num", type=int, default=32)
    # parser.add_argument("--channel_2_num", type=int, default=64)
    # parser.add_argument("--conv_size", type=int, default=5)
    # parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--hidden_size_1", type=int, default=8)
    parser.add_argument("--hidden_size_2", type=int, default=8)
    parser.add_argument("--hidden_size_3", type=int, default=8)
    # parser.add_argument("--learning_rate", type=float, default=1e-4)
    # parser.add_argument("--batch_num", type=int, default=2000)
    # parser.add_argument("--batch_size", type=int, default=32)

    args, _ = parser.parse_known_args()
    return args



if __name__ == "__main__":
    '''@nni.get_next_parameter()'''
    try:
        main(vars(get_params()))
    except Exception as exception:
        logger.exception(exception)
        raise
