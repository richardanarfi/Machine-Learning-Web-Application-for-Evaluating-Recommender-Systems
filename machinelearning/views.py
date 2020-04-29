from django.shortcuts import render
import sys
from subprocess import run,PIPE

from django.http import HttpResponse
from matplotlib import pylab
from pylab import *
import PIL, PIL.Image, io as StringIO
import csv
import re
import mimetypes


def NN_model(request):
    print(request.POST)
    train_test = request.POST.get('train_test')
    activation_function = request.POST.get('activation_function')
    dropout = int(request.POST.get('dropout'))/100
    n_epochs = request.POST.get('n_epochs')
    gru_layers = request.POST.get('gru_layers')
    loss = request.POST.get('loss')
    final_act = request.POST.get('final_act')
    test_epochs = int(n_epochs) - 1
    if train_test == 'Train_Test':
        train = 1
        test_epochs = int(n_epochs) - 1
        test = test_epochs
    elif train_test == 'Train':
        train = 1
        test = 0
    elif train_test == 'Evaluate':
        train = 0
        test = test_epochs
    else:
        train = 0
        test = 0
    input1 = '--train=' + str(train)
    input2 = '--epoch=' + str(n_epochs)
    input3 = '--test=' + str(test)
    input4 = '--hidden_act=' + activation_function
    input5 = '--dropout=' + str(dropout)
    input6 = '--loss=' + loss
    input7 = '--final_act=' + final_act
    #train = False
    #test = False
    if train:
        run([sys.executable, 'machinelearning/NN_model/main.py', input1, input2, input3, input4, input5, input6, input7], shell=False, stdout=PIPE)
    if train and test:
        input1 = '--train=0'
        run([sys.executable, 'machinelearning/NN_model/main.py', input1, input2, input3, input4, input5, input6, input7], shell=False, stdout=PIPE)
    elif test:
        run([sys.executable, 'machinelearning/NN_model/main.py', input1, input3], shell=False, stdout=PIPE)

    out1 = open('train_results.txt', 'r').readlines()
    out2 = open('test_results.txt', 'r').readlines()
    out3 = []
    out4 = []
    out5 = []
    out6 = []
    recall1 = []
    recall2 = []
    mrr1 = []
    mrr2 = []

    with open('train_results.txt') as graphsource:
        next(graphsource)
        next(graphsource)
        allrawdata = []
        for line in graphsource:
            tmp = re.findall(r'\d+(?:\.\d+)?', line)
            allrawdata.append(tmp)
	
    for i in range(len(allrawdata)):
        ep = int(allrawdata[i][0], 10)
        step = int(allrawdata[i][1], 10)
        lr = float(allrawdata[i][2])
        loss = float(allrawdata[i][3])
        out3.append(ep)
        out4.append(step)
        out5.append(lr)
        out6.append(loss)
		
    with open('test_results.txt') as graphsource:
        next(graphsource)
        next(graphsource)
        next(graphsource)
        next(graphsource)
        allrawdata = []
        for line in graphsource:
            tmp = re.findall(r'\d+(?:\.\d+)?', line)
            allrawdata.append(tmp)
			
    for i in range(len(allrawdata)):
        if i % 2 == 1:
            tr1 = int(allrawdata[i][0], 10)
            tr2 = float(allrawdata[i][1])
            recall1.append(tr1)
            recall2.append(tr2)
        if i % 2 == 0:
            tr3 = int(allrawdata[i][0], 10)
            tr4 = float(allrawdata[i][1])
            mrr1.append(tr3)
            mrr2.append(tr4)
			
    return render(request, 'NN.html', {'data_train': out1, 'data_test': out2, 'data_epochs': out3, 'data_rate': out4, 'data_lr': out5, 'data_loss': out6, 'recall1': recall1, 'recall2': recall2, 'mrr1': mrr1, 'mrr2': mrr2})


def matrixFactorization(request):
    print(request.POST)
    input1 = request.POST.get('dataset')
    input2 = request.POST.get('train_perc')
    input3 = request.POST.get('n_sim_users')
    input4 = request.POST.get('n_rec')
    print(input1)
    print(input2)
    print(input3)
    print(input4)
    #input1 = "ratings100k.dat"
    #input2 = "0.7"
    #input3 = "20"
    #input4 = "10"
    #out= run([sys.executable,'//cf.py',inp],shell=False,stdout=PIPE)
    run([sys.executable,"machinelearning/cf.py",input1,input2,input3,input4],shell=False,stdout=PIPE)
    #print(out)
    file = open("results.txt","r").readlines()
    return render(request,'MF.html',{'data':file})