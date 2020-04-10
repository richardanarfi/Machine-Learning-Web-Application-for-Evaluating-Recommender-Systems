# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

import model
import evaluation

from sklearn.model_selection import train_test_split

PATH_TO_TRAIN = 'C:/Users/bkwap/Desktop/Web-Application-for-Evaluating-Recommender-Systems-Machine-Learning-Models/machinelearning/NN_model/ratings.csv' #/PATH/TO/rsc15_train_full.txt'
#PATH_TO_TEST = 'e:/sundog-consult/Udemy/RecSys/GRU4Rec_TensorFlow-master/ratings.csv' #'/PATH/TO/rsc15_test.txt'

class Args():
    is_training = False
    layers = 2
    rnn_size = 100
    n_epochs = 10
    batch_size = 128
    dropout_p_hidden=1
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

def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
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
    command_line = parseArgs()
    data = pd.read_csv(PATH_TO_TRAIN, dtype={'movieId': np.int64})
    valid = data.iloc[90000:, :]
    data = data.iloc[:90000, :]
    #valid = pd.read_csv(PATH_TO_TEST, dtype={'movieId': np.int64})
    #data, valid = train_test_split(data, random_state=42)
    args = Args()
    args.n_items = len(data['movieId'].unique())
    args.layers = command_line.layer
    args.rnn_size = command_line.size
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout
    print(args.dropout_p_hidden)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        gru = model.GRU4Rec(sess, args)
        if args.is_training:
            output = open('train_results.txt', 'w')
            gru.fit(data)
            output.close()
        else:
            test_output = open('test_results.txt', 'w')
            print("\n\nEvaluating Model....\n", file = test_output)
            res = evaluation.evaluate_sessions_batch(gru, data, data, 5)
            print('Recall@5: {}'.format(res[0][0]), file = test_output)
            print('MRR@5: {}'.format(res[1][0]), file = test_output)
            #res = evaluation.evaluate_sessions_batch(gru, data, data, 10)
            print('Recall@10: {}'.format(res[0][1]), file = test_output)
            print('MRR@10: {}'.format(res[1][1]), file = test_output)
            #res = evaluation.evaluate_sessions_batch(gru, data, data, 20)
            print('Recall@20: {}'.format(res[0][2]), file = test_output)
            print('MRR@20: {}'.format(res[1][2]), file = test_output)
            #res = evaluation.evaluate_sessions_batch(gru, data, data, 50)
            print('Recall@50: {}'.format(res[0][3]), file = test_output)
            print('MRR@50: {}'.format(res[1][3]), file = test_output)
            test_output.close()
    
