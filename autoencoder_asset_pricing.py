import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
from sklearn.metrics import mean_absolute_error
import time
from copy import deepcopy




class acutoencoder_pricing(nn.Module):
    def __init__(self,port_nums,stock_nums,factor_nums,k,drop_prob):
        super(acutoencoder_pricing,self).__init__()
        ###beta encoder
        self.fc1=nn.Linear(port_nums,k)
        self.fc2=nn.Linear(stock_nums,factor_nums)
        self.drop_out=nn.Dropout(drop_prob)
        self.bn1=nn.BatchNorm1d(k)
        self.bn2=nn.BatchNorm1d(factor_nums)
        self.model1=nn.Sequential(self.fc1,self.bn1,nn.ReLU(),self.drop_out)
        self.model2=nn.Sequential(self.fc2,self.bn2,nn.ReLU(),self.drop_out)
        
        
        
    def forward(self):
        betas=self.model1.view(-1,self.k)
        factors=self.model2.view(self.factor_nums,1)
        return torch.mm(betas,factors)
    
    
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")

"""port_nums: 주식 종목수 x 팩터
    k: 주식 종목수 x 축소 후 latent factor"""
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.port_nums=317*81
args.stock_nums=317
args.factor_nums=32
args.k=317*32

# ====== Regularization ======= #
args.l2 = 0.00001
args.drop_prob = 0.0
args.use_bn = True
args.batch_size=317 ####  주식 종목수 만큼 배치사이즈를 정함
# ====== Optimizer & Training ====== #
args.optim = 'ADAM' #'RMSprop' #SGD, RMSprop, ADAM...
args.lr = 0.0001
args.epoch = 2
name_var1 = 'lr'
name_var2 = 'n_layers'
list_var1 = [0.001, 0.0001, 0.00001]
list_var2 = [1,2,3]
df=pd.read_csv('data.csv',index_col=0,parse_dates=True)
y_input = df.returns.to_numpy()
y= y_input.reshape(1, len(df.returns))
x= df.iloc[:,1:].to_numpy() ####returns 제외



num_workers=0
def metric(y_pred, y_true):
    perc_y_pred = np.exp(y_pred.cpu().detach().numpy())
    perc_y_true = np.exp(y_true.cpu().detach().numpy())
    mae = mean_absolute_error(perc_y_true, perc_y_pred, multioutput='raw_values')
    return mae*100





def train(model, partition, partition2,optimizer,criterion, args):
    trainloader_x = DataLoader(partition['train'], 
                             batch_size=args.batch_size, 
                             shuffle=True, num_workers=num_workers)
    trainloader_y = DataLoader(partition2['train'], 
                             batch_size=args.batch_size, 
                             shuffle=True, num_workers=num_workers)
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    train_acc = 0.0
    train_loss = 0.0
    for X, y in zip(trainloader_x,trainloader_y):

        X = X.float().to(args.device)
        y_true = y.float().to(args.device)
        #print(torch.max(X[:, :, 3]), torch.max(y_true))

        model.zero_grad()
        optimizer.zero_grad()
        #model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

        y_pred = model(y=y_true.float(),x=X.float())
        loss = criterion(y_pred,y_true.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += metric(y_pred, y_true)[0]

    train_loss = train_loss / len(trainloader_x)
    train_acc = train_acc / len(trainloader_x)
    return model, train_loss, train_acc




def validate(model, partition, partition2, criterion, args):
    valloader_x = DataLoader(partition['val'], 
                           batch_size=args.batch_size, 
                           shuffle=False, drop_last=True)
    valloader_y = DataLoader(partition2['val'], 
                           batch_size=args.batch_size, 
                           shuffle=False, drop_last=True)
    model.eval()

    val_acc = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for X, y in zip(valloader_x,valloader_y):

            X = X.float().to(args.device)
            y_true = y.float().to(args.device)
            #model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

            y_pred = model(y=y_true.float(),x=X.float())
            loss = criterion(y_pred,y.float())

            val_loss += loss.item()
            val_acc += metric(y_pred, y_true)[0]

    val_loss = val_loss / len(valloader_x)
    val_acc = val_acc / len(valloader_y)
    return val_loss, val_acc





def test(model, partition, partition2, args):
    testloader_x = DataLoader(partition['test'], 
                           batch_size=args.batch_size, 
                           shuffle=False, drop_last=True)
    testloader_y = DataLoader(partition2['test'], 
                           batch_size=args.batch_size, 
                           shuffle=False, drop_last=True)
    model.eval()

    test_acc = 0.0
    with torch.no_grad():
        for X, y in enumerate(testloader_x,testloader_y):

            X = X.float().to(args.device)
            y_true = y.float().to(args.device)
            #model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

            y_pred = model(y=y_true.float(),x=X.float())
            test_acc += metric(y_pred, y_true)[0]

    test_acc = test_acc / len(testloader_x)
    return test_acc   






def experiment(partition,partition2, args):
  
    model = model(args.port_nums,args.stock_nums,args.factor_nums,args.k,args.drop_prob)
    model.cuda()

    criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
        
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        ts = time.time()
        model, train_loss, train_acc = train(model, partition, partition2, optimizer, criterion, args)
        val_loss, val_acc = validate(model, partition, partition2, criterion, args)
        te = time.time()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print('Epoch {}, Acc(train/val): {:2.2f}/{:2.2f}, Loss(train/val) {:2.2f}/{:2.2f}. Took {:2.2f} sec'.format(epoch, train_acc, val_acc, train_loss, val_loss, te-ts))
        
    test_acc = test(model, partition,partition2,args)    
    
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['train_accs'] = train_accs
    result['val_accs'] = val_accs
    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc
    return vars(args), result

temp1='2000-01-01'
temp2='2019-01-01'
temp3='2021-01-01'
temp4='2022-11-15'
train_x = x.loc[temp1:temp2]
val_x=x.loc[temp2:temp3]
test_x=x.loc[temp3:temp4]
train_y = y.loc[temp1:temp2]
val_y=y.loc[temp2:temp3]
test_y=y.loc[temp3:temp4]


partition = {'train': train_x, 'val':val_x, 'test':test_x}
partition2={'train': train_y, 'val':val_y, 'test':test_y}
for var1 in list_var1:
    for var2 in list_var2:
        setattr(args, name_var1, var1)
        setattr(args, name_var2, var2)
        print(args)
                
        setting, result = experiment(partition, deepcopy(args))
        #save_exp_result(setting, result)