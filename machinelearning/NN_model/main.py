# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import os
import argparse
import time
import tensorflow as tf
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
import model
import evaluation



PATH_TO_TRAIN = 'machinelearning/NN_model/ratings.csv' 


class ARGS():
    '''
    Define arguments
    '''
	# pylint: disable=too-many-instance-attributes
	# pylint: disable=too-few-public-methods
    is_training = False
    layers = 2
    rnn_size = 100
    n_epochs = 10
    batch_size = 128
    dropout_p_hidden = 1
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'userId'
    item_key = 'movieId'
    time_key = 'timestamp'
    grad_cap = 0
    test_model = 2
    checkpoint_dir = './checkpoint'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1

def parse_args():
    '''
    Argument parser
    '''
    parser = argparse.ArgumentParser(description='GRU4Rec ARGS')
    parser.add_argument('--layer', default=2, type=int)
    parser.add_argument('--size', default=128, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', default=2, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)
    return parser.parse_args()


if __name__ == '__main__':
    COMMAND_LINE = parse_args()
    DATA = pd.read_csv(PATH_TO_TRAIN, dtype={'movieId': np.int64})
    VALID = DATA.iloc[90000:, :]
    DATA = DATA.iloc[:90000, :]
    #VALID = pd.read_csv(PATH_TO_TEST, dtype={'movieId': np.int64})
    #DATA, VALID = train_test_split(DATA, random_state=42)
    ARGS = ARGS()
    ARGS.n_items = len(DATA['movieId'].unique())
    ARGS.layers = COMMAND_LINE.layer
    ARGS.rnn_size = COMMAND_LINE.size
    ARGS.n_epochs = COMMAND_LINE.epoch
    ARGS.learning_rate = COMMAND_LINE.lr
    ARGS.is_training = COMMAND_LINE.train
    ARGS.test_model = COMMAND_LINE.test
    ARGS.hidden_act = COMMAND_LINE.hidden_act
    ARGS.final_act = COMMAND_LINE.final_act
    ARGS.loss = COMMAND_LINE.loss
    ARGS.dropout_p_hidden = 1.0 if ARGS.is_training == 0 else COMMAND_LINE.dropout
    print(ARGS.dropout_p_hidden)
    if not os.path.exists(ARGS.checkpoint_dir):
        os.mkdir(ARGS.checkpoint_dir)
    GPU_CONFIG = tf.ConfigProto()
    #GPU_CONFIG.gpu_options.allow_growth = True
    with tf.Session(config=GPU_CONFIG) as sess:
        GRU = model.GRU4Rec(sess, ARGS)
        START_TIME = time.time()
        if ARGS.is_training:
            OUTPUT = open('train_results.txt', 'w')
            GRU.fit(DATA)
            OUTPUT.close()
            TRAINING_TIME = time.time()
            print("Training time =", TRAINING_TIME - START_TIME, "seconds")
        else:
            TEST_OUTPUT = open('test_results.txt', 'w')
            print("\n\nEvaluating Model....\n", file=TEST_OUTPUT)
            RES = evaluation.evaluate_sessions_batch(GRU, DATA, DATA)
            print('Recall@1: {}'.format(RES[0][0]), file=TEST_OUTPUT)
            print('MRR@1: {}'.format(RES[1][0]), file=TEST_OUTPUT)
            print('Recall@2: {}'.format(RES[0][1]), file=TEST_OUTPUT)
            print('MRR@2: {}'.format(RES[1][1]), file=TEST_OUTPUT)
            print('Recall@5: {}'.format(RES[0][2]), file=TEST_OUTPUT)
            print('MRR@5: {}'.format(RES[1][2]), file=TEST_OUTPUT)
            #RES = evaluation.evaluate_sessions_batch(GRU, DATA, DATA, 10)
            print('Recall@10: {}'.format(RES[0][3]), file=TEST_OUTPUT)
            print('MRR@10: {}'.format(RES[1][3]), file=TEST_OUTPUT)
            #RES = evaluation.evaluate_sessions_batch(GRU, DATA, DATA, 20)
            print('Recall@20: {}'.format(RES[0][4]), file=TEST_OUTPUT)
            print('MRR@20: {}'.format(RES[1][4]), file=TEST_OUTPUT)
            #RES = evaluation.evaluate_sessions_batch(GRU, DATA, DATA, 50)
            print('Recall@50: {}'.format(RES[0][5]), file=TEST_OUTPUT)
            print('MRR@50: {}'.format(RES[1][5]), file=TEST_OUTPUT)
            print('Recall@100: {}'.format(RES[0][6]), file=TEST_OUTPUT)
            print('MRR@100: {}'.format(RES[1][6]), file=TEST_OUTPUT)
            #print('Precision@1: {}'.format(RES[2][0]), file = TEST_OUTPUT)
            #print('Precision@2: {}'.format(RES[2][1]), file = TEST_OUTPUT)
            #print('Precision@5: {}'.format(RES[2][2]), file = TEST_OUTPUT)
            #print('Precision@10: {}'.format(RES[2][3]), file = TEST_OUTPUT)
            #print('Precision@20: {}'.format(RES[2][4]), file = TEST_OUTPUT)
            #print('Precision@50: {}'.format(RES[2][5]), file = TEST_OUTPUT)
            #print('Precision@100: {}'.format(RES[2][6]), file = TEST_OUTPUT)
            TEST_OUTPUT.close()
        END_TIME = time.time()
        print("Time elapsed =", END_TIME - START_TIME, "seconds")
