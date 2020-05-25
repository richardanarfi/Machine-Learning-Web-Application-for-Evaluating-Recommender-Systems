# -*- coding: utf-8 -*-
"""
Created on Feb 27 2017
Author: Weiping Song
"""
import pandas as pd
import numpy as np

MRR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
RECALL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
BATCHSIZE = [128]
EVALPOINTCOUNT = [0]

def cut_off():
    ''' Cut-off value (i.e. the length of the recommendation list '''
    cut_off1 = 1
    cut_off2 = 2
    cut_off3 = 5
    cut_off4 = 10
    cut_off5 = 20
    cut_off6 = 50
    cut_off7 = 100
    return (cut_off1, cut_off2, cut_off3, cut_off4, cut_off5, cut_off6, cut_off7)

def itemidmapx(train_data, test_data):
    ''' Build itemidmap from train data '''
    itemids = train_data['movieId'].unique()
    itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)
    test_data.sort_values(['userId', 'timestamp'], inplace=True)
    offset_sessions = np.zeros(test_data['userId'].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby('userId').size().cumsum()
    #evalutation_point_count = 0
    itemx = (itemidmap, offset_sessions)
    return itemx

def evaluate_sessions_batch(model, train_data, test_data):
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.

    Parameters
    --------
    model : A trained GRU4Rec model.
    train_data : Contains the transactions of the train set.
    In  evaluation phrase, this is used to build item-to-id map.
    test_data : It contains the transactions of the test set.
    It has one  column for session IDs, one for item IDs
    and one for the timestamp of  the events (unix timestamps).
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list;
        N for recall@N and MRR@N). Defauld value is 20.
    BATCHSIZE[0] : int
        Number of events bundled into a batch during evaluation.
        Speeds up evaluation.
        If it is set high, the memory consumption increases. Default value is 100.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)
    '''
    # (cut_off1, cut_off2, cut_off3, cut_off4, cut_off5, cut_off6, cut_off7) = \
    #(1, 2, 5, 10, 20, 50, 100)
    #(session_key, item_key, time_key) = ('userId', 'movieId', 'timestamp')
    model.predict = False
    # Build itemidmap from train data.
    #itemids = train_data['movieId'].unique()
    #itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)
    #test_data.sort_values(['userId', 'timestamp'], inplace=True)
    #offset_sessions = np.zeros(test_data['userId'].nunique()+1, dtype=np.int32)
    #offset_sessions[1:] = test_data.groupby('userId').size().cumsum()
    #evalutation_point_count = 0
    itemx = itemidmapx(train_data, test_data)
    #mrr, recall, precision = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #mrr, recall = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
    #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # precision_counter = [0, 0, 0, 0, 0, 0, 0]
    if len(itemx[1]) - 1 < BATCHSIZE[0]:
        BATCHSIZE[0] = len(itemx[1]) - 1
    iters = np.arange(BATCHSIZE[0]).astype(np.int32)
    maxiter = iters.max()
    start = itemx[1][iters]
    end = itemx[1][iters+1]
    in_idx = np.zeros(BATCHSIZE[0], dtype=np.int32)
    #np.random.seed(42)
    while True:
        valid_mask = iters >= 0
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask]
        #minlen = (end[valid_mask]-start_valid).min()
        in_idx[valid_mask] = test_data['movieId'].values[start_valid]
        for i in range((end[valid_mask]-start_valid).min()-1):
            #out_idx = test_data['movieId'].values[start_valid+i+1]
            preds = model.predict_next_batch(iters, in_idx, itemx[0], BATCHSIZE[0])
            preds.fillna(0, inplace=True)
            in_idx[valid_mask] = test_data['movieId'].values[start_valid+i+1]
            ranks = (preds.values.T[valid_mask].T > \
            np.diag(preds.ix[in_idx].values)[valid_mask]).sum(axis=0) + 1
            #rank1_ok = ranks <= cut_off()[0]
            RECALL[0] += (ranks <= cut_off()[0]).sum()
            #precision[0] += rank1_ok.sum()
            #precision_counter[0] += cut_off1
            MRR[0] += (1.0 / ranks[(ranks <= cut_off()[0])]).sum()
            #rank2_ok = ranks <= cut_off()[1]
            RECALL[1] += (ranks <= cut_off()[1]).sum()
            #precision[1] += rank2_ok.sum()
            #precision_counter[1] += cut_off2
            MRR[1] += (1.0 / ranks[ranks <= cut_off()[1]]).sum()
            #rank3_ok = ranks <= cut_off()[2]
            RECALL[2] += (ranks <= cut_off()[2]).sum()
            #precision[2] += rank3_ok.sum()
            #precision_counter[2] += cut_off3
            MRR[2] += (1.0 / ranks[ranks <= cut_off()[2]]).sum()
            #rank4_ok = ranks <= cut_off()[3]
            RECALL[3] += (ranks <= cut_off()[3]).sum()
            #precision[3] += rank4_ok.sum()
            #precision_counter[3] += cut_off4
            MRR[3] += (1.0 / ranks[ranks <= cut_off()[3]]).sum()
            #rank5_ok = ranks <= cut_off()[4]
            RECALL[4] += (ranks <= cut_off()[4]).sum()
            #precision[4] += rank5_ok.sum()
            #precision_counter[4] += cut_off5
            MRR[4] += (1.0 / ranks[ranks <= cut_off()[4]]).sum()
            #rank6_ok = ranks <= cut_off()[5]
            RECALL[5] += (ranks <= cut_off()[5]).sum()
            #precision[5] += rank6_ok.sum()
            #precision_counter[5] += cut_off6
            MRR[5] += (1.0 / ranks[ranks <= cut_off()[5]]).sum()
            #rank7_ok = ranks <= cut_off()[6]
            RECALL[6] += (ranks <= cut_off()[6]).sum()
            #precision[6] += rank7_ok.sum()
            #precision_counter[6] += cut_off7
            MRR[6] += (1.0 / ranks[ranks <= cut_off()[6]]).sum()
            EVALPOINTCOUNT[0] += len(ranks)
        start = start + (end[valid_mask]-start_valid).min() - 1
        #mask = np.arange(len(iters))[(valid_mask) & (end-start <= 1)]
        for idx in np.arange(len(iters))[(valid_mask) & (end-start <= 1)]:
            maxiter += 1
            if maxiter >= len(itemx[1])-1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = itemx[1][maxiter]
                end[idx] = itemx[1][maxiter+1]
    #for i in range(len(precision)):
        #precision[i] = precision[i]/precision_counter[i]
    return [x/EVALPOINTCOUNT[0] for x in RECALL], \
    [y/EVALPOINTCOUNT[0] for y in MRR]#, precision
