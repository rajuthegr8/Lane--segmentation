## Contains functions for training the Model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

def train(net, train_loader, criterion, optimizer, num_epoch, lr, PATH, with_graph=False, save_=False):
    # criterion = nn.CrossEntropyLoss().to(device)

    j = 0
    j_list = []
    loss_list = []
    for epoch in range(0, num_epoch):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, label = data

            inputs = inputs.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
 
            loss = criterion(outputs, label.long())
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
        running_loss /= len(train_loader)
        print('Epoch No => ', epoch)
        print('running loss => ', running_loss)
        j+=1
        if with_graph:
            j_list.append(j)
            loss_list.append(running_loss)
            plt.plot(j_list, loss_list)
            plt.pause(0.00001)

        if save_ and epoch%10==0:
            torch.save(net, '/content/drive/My Drive/Models/DeepLabv3_10epoch_lr0.001_currepoch'+str(epoch)+'.pt')

    #     print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(epoch+1, running_loss, train_accuracy, test_accuracy))
    torch.save(net, PATH)
    return net