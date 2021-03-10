import cv2
import numpy as np
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn, optim

def acc_plt_show(num_epochs,training_accuracy,validation_accuracy,LR,save_file):
    plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
    plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
    plt.title('Training & Validation accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("{}/{}_{}_acc.jpg".format(save_file,num_epochs,LR))
    plt.show()

def loss_plt_show(num_epochs,training_loss,validation_loss,LR,save_file):
    plt.plot(range(num_epochs), training_loss, 'b-', label='Training_loss')
    plt.plot(range(num_epochs), validation_loss, 'g-', label='validation_loss')
    plt.title('Training & Validation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("{}/{}_{}_loss.jpg".format(save_file,num_epochs,LR))
    plt.show()

def plt_show(image):
    
    plt.imshow(image)
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()

def get_dataset(BATCH_SIZE):
    
    path="save_numpy"
    
    train_image_label= np.load(os.path.join(path,'train_image_label.npy'),allow_pickle=True)
    train_image_label = train_image_label.tolist()

    val_image_label= np.load(os.path.join(path,'val_image_label.npy'),allow_pickle=True)
    val_image_label = val_image_label.tolist()
    
    class_name= np.load(os.path.join(path,'class_name.npy'),allow_pickle=True)
    
    train_Loader = DataLoader(dataset=train_image_label,batch_size=BATCH_SIZE,shuffle=True)
    val_Loader = DataLoader(dataset=val_image_label,batch_size=4,shuffle=True)
    
    return train_Loader,val_Loader,class_name

if __name__ == "__main__":

    
    save_file="save_resnet"
    if not os.path.isdir(save_file):
        os.mkdir(save_file)
        
    EPOCHS=70
    LR=0.0125
    batch_size=24
    
    train_Loader,val_Loader,class_name = get_dataset(batch_size)
    #train_Loader,class_name = get_dataset(batch_size)
    num_class=len(class_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = models.resnet50(pretrained=True)
    
    net.fc = torch.nn.Sequential(torch.nn.Linear(2048, 1000),
                                 torch.nn.Linear(1000, 500),
                                 torch.nn.Linear(500, 100),
                                 torch.nn.Linear(100, num_class),
                                 )
    #net.fc = torch.nn.Linear(2048, num_class)
    
    for i,(name, parma) in enumerate(net.named_parameters()):
        if(i<129):
            parma.requires_grad=False
        #if(int(name.split('.')[1])==2 and name.split('.')[0]=='layer4'):
        #    parma.requires_grad=False
        print(i,name,parma.requires_grad)#name.split('.')
        
    print(net)
    
    #optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    
    
    optimizer = torch.optim.SGD([{'params':net.fc.parameters()},
                                 {'params':net.layer4.parameters()}], 
                                lr=LR, momentum=0.9, weight_decay=5e-4)
    '''
    optimizer = torch.optim.Adam([{'params':net.fc.parameters()},
                                 {'params':net.layer4[2].parameters()}], 
                                 lr=LR)
    '''
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=10,factor=0.5)
    net = net.to(device)
    #net.train()
    
    loss_func = torch.nn.CrossEntropyLoss()
    #loss_func = torch.nn.BCELoss()
    
    training_loss,validation_loss=[],[]
    training_accuracy,validation_accuracy =[],[]
    
    best_acc,best_epoch=0,0
    
    for epoch in range(EPOCHS):
        net.train()
        train_loss_reg,total_train,step_count,correct_train =0.0, 0,0,0
        #if(epoch in [50,70]):
        #    learn_rate=learn_rate/10
        #    print("****************learn_rate=",learn_rate,"*****************")
        #    optimizer = torch.optim.Adam(net.classifier.parameters(),lr=learn_rate)
        
        for step, (batch_x,label_y) in enumerate(train_Loader):
            batch_x = torch.FloatTensor(batch_x.type(torch.FloatTensor)/255.0)
            #batch_x = torch.FloatTensor((batch_x.type(torch.FloatTensor)-127.5)/128.0)
            label_y = torch.LongTensor(label_y)
            h,w,c= batch_x.shape[1],batch_x.shape[2],batch_x.shape[3]
            input_shape = (-1,c,h,w)
            train = Variable(batch_x.view(input_shape)).to(device)
            labels = Variable(label_y).to(device)
            outputs = net(train)
            train_loss = loss_func(outputs,labels)
            
            optimizer.zero_grad()               # clear gradients for this training step
            train_loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            
            train_loss_reg +=train_loss.cpu().data
            step_count += 1
            
            ans=torch.max(outputs,1)[1].squeeze()
            total_train += len(labels)
            correct_train += (ans.cpu() == labels.cpu()).float().sum()
            
            print("Epoch:{}/{} Step:{}/{} Train_loss:{:1.3f} ".format(epoch+1,EPOCHS,step+1,len(train_Loader),train_loss))
            
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        
        #print(step,step_count)
        avg_train_loss = train_loss_reg/step_count
        training_loss.append(avg_train_loss)
        print("{}[Epoch:{}/{}  Avg_train_loss:{:1.3f} Acc_train:{:3.2f}%]".format(("*"*30),epoch+1,EPOCHS,avg_train_loss,train_accuracy))#loss.item()
        
        with torch.no_grad():
            net.eval()
            val_loss_reg,total_val,step_count,correct_val =0.0, 0,0,0
            for step, (batch_x,label_y) in enumerate(val_Loader):
                batch_x = torch.FloatTensor(batch_x.type(torch.FloatTensor)/255.0)
                label_y = torch.LongTensor(label_y)
                h,w,c= batch_x.shape[1],batch_x.shape[2],batch_x.shape[3]
                input_shape = (-1,c,h,w)
                val = Variable(batch_x.view(input_shape)).to(device)
                labels = Variable(label_y).to(device)
                outputs = net(val)
                val_loss = loss_func(outputs,labels)
    
                val_loss_reg +=val_loss.cpu().data
                step_count += 1
                
                ans=torch.max(outputs,1)[1].squeeze()
                total_val += len(labels)
                correct_val += (ans.cpu() == labels.cpu()).float().sum()
            
            val_accuracy = 100 * correct_val / float(total_val)
            validation_accuracy.append(val_accuracy)
            
            avg_val_loss = val_loss_reg/step_count
            validation_loss.append(avg_val_loss)
            print("{}[Epoch:{}/{}  Avg_val_loss:{:1.3f} Acc_val:{:3.2f}]".format(("*"*30),epoch+1,EPOCHS,avg_val_loss,val_accuracy))
            if(val_accuracy>best_acc):
                    best_acc=val_accuracy
                    torch.save(net.state_dict(), '{}/save_net_best_{}.{}_{}.pkl'.format(save_file,epoch+1,EPOCHS,LR))
            torch.cuda.empty_cache()
        scheduler.step(avg_val_loss)
        lr = optimizer.param_groups[0]['lr']
        print("LR:{}".format(lr))

    loss_plt_show(EPOCHS,training_loss,validation_loss,LR,save_file)
    acc_plt_show(EPOCHS,training_accuracy,validation_accuracy,LR,save_file)
    torch.save(net.state_dict(), '{}/net_{}_{}.pkl'.format(save_file,EPOCHS,LR))
    

