import sys
from subprocess import run, PIPE

from django.shortcuts import render


def NN_model(request):
    print(request.POST)
    train_test = request.POST.get('train_test')
    activation_function = request.POST.get('activation_function')
    dropout = int(request.POST.get('dropout')) / 100
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
    # train = False
    # test = False
    if train:
        run([sys.executable, 'machinelearning/NN_model/main.py', input1, input2, input3, input4, input5, input6,
             input7], shell=False, stdout=PIPE)
    if train and test:
        input1 = '--train=0'
        run([sys.executable, 'machinelearning/NN_model/main.py', input1, input2, input3, input4, input5, input6,
             input7], shell=False, stdout=PIPE)
    elif test:
        run([sys.executable, 'machinelearning/NN_model/main.py', input1, input3], shell=False, stdout=PIPE)

    out1 = open('train_results.txt', 'r').readlines()

    out2 = open('test_results.txt', 'r').readlines()

    return render(request, 'NN.html', {'data_train': out1, 'data_test': out2})


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
    # input1 = "ratings100k.dat"
    # input2 = "0.7"
    # input3 = "20"
    # input4 = "10"
    # out= run([sys.executable,'//cf.py',inp],shell=False,stdout=PIPE)
    run([sys.executable, "machinelearning/cf.py", input1, input2, input3, input4], shell=False, stdout=PIPE)
    # print(out)
    file = open("results.txt", "r").readlines()
    graph_data = open("graph_data.txt", "r").readlines()
    graph_data = [x.strip("\n").split('=') for x in graph_data]
    graph_prec = open("graph_data2.txt","r").readlines()
    graph_prec = [x.strip("\n").split('=') for x in graph_prec]
    graph_rec = open("graph_data3.txt","r").readlines()
    graph_rec = [x.strip("\n").split('=') for x in graph_rec]
    data2 = []
    data3 = []
    data4 = []

    for item in graph_data:
        data2.append(float(item[1]))

    for item2 in graph_prec:
        data3.append(float(item2[1]))

    for item3 in graph_rec:
        data4.append(float(item3[1]))


    graph_set1 = data2[:3]
    graph_set2 = data2[3:6]
    graph_set3 = data2[6:]

    graph_set4 = data3[:3]
    graph_set5 = data4[:3]


    user1 = str(int(input3) - 5) + ' users'
    user2 = input3 + ' users'
    user3 = str(int(input3) + 5) + 'users'


    return render(request, 'MF.html',
                  {'data': file, 'graph_data1': graph_set1, 'graph_data2': graph_set2, 'graph_data3': graph_set3
                   ,'user_data1': user1, 'user_data2': user2, 'user_data3': user3,'prec': graph_set4,'rec':graph_set5})

