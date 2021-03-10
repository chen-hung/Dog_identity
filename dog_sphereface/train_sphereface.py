import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os 
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import time
from PIL import Image, ImageDraw
from torch.autograd import Variable
from sphereface import AngleSoftmax
from module.sphere_face import *
from module.loss import *

torch.backends.cudnn.bencmark = True

from net_sphere import sphere20a,AngleLoss#,SphereProduct
#from sphereface_pytorch.matlab_cp2tform import get_similarity_transform_for_cv2
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


matplotlib_axes_logger.setLevel('ERROR')

    
def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)



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
    
   
    
    torch.autograd.set_detect_anomaly(True)
    
    epoch=70
    learn_rate=0.0125
    batch_size=64
    
    
    train_Loader,val_Loader,class_name=get_dataset(batch_size)
    #train_Loader,class_name=get_dataset(batch_size)
    num_class=len(class_name)

    loss_epoch,loss_error=[],[]
    #plt_x,plt_y,plt_color=[],[],[]
    
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SphereFace20(num_classes=num_class, feat_dim=512).to(device)
    #net= sphere20a(classnum=num_class).to(device)
    #net.load_state_dict(torch.load('0_save_sphere/sphere20a_60.pth'))
    # net.load_state_dict(torch.load('sphere20a_0.pth'))
    #net.train()
    
    optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=10,factor=0.5)
    #optimizer = torch.optim.SGD(net.parameters(),lr=learn_rate, momentum=0.5, weight_decay=1e-5, nesterov=True)
    
    #loss_func = AngleLoss()
    #loss_func = SphereProduct(512,num_class) #AngleSoftmax(512,num_class)
    loss_func = AngularSoftmaxWithLoss()
    
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    
    best_acc,best_epoch=0,0
    
    save_file="save_sphere_part2"
    if not os.path.isdir(save_file):
        os.mkdir(save_file)
    
    for i in range(epoch):
        net.train()
        #print("..................learn_rate="+str(learn_rate))
        train_loss_reg = 0.0
        total_train = 0
        step_count = 0
        correct_train = 0
        #scatter_x , scatter_y,scatter_color =[],[],[]
        for step, (batch_x,label_y) in enumerate(train_Loader):
            batch_x = torch.FloatTensor((batch_x.type(torch.FloatTensor)-127.5)/128.0)#(train_data-127.5)/128.0
            label_y = torch.LongTensor(label_y)
            h,w,c= batch_x.shape[1],batch_x.shape[2],batch_x.shape[3]
            input_shape = (-1,c,h,w)
            train = batch_x.view(input_shape).to(device)#,volatile=True)
            labels = label_y.to(device)#,volatile=True)
            #outputs,features_512= net(train) 
            features_512,outputs= net(train)
            
            #feat,train_loss = loss_func.forward(features_512.cpu(),labels.cpu())
            train_loss = loss_func(outputs,labels)
            
            
            optimizer.zero_grad()               # clear gradients for this training step
            train_loss.backward()#(retain_graph=True)                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            
            train_loss_reg +=train_loss.item()
            step_count += 1
            
            
            ans=[]
            ans_cos,ans_phi=torch.max(outputs[0],1)[1].squeeze(),torch.max(outputs[1],1)[1].squeeze()
            if((ans_cos == ans_phi).all()):
                ans = ans_cos
            total_train += len(labels)
            correct_train += (ans.cpu() == labels.cpu()).float().sum()
            #ans = torch.max(feat,-1)[1].squeeze()
            #total_train += len(labels)
            #correct_train += (ans.cpu() == labels.cpu()).float().sum()
            print("Epoch:{}/{} Step:{}/{} Train_loss:{:1.3f} ".format(i+1,epoch,step+1,len(train_Loader),train_loss))
            torch.cuda.empty_cache()
            
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        
        avg_train_loss = train_loss_reg/step_count
        training_loss.append(avg_train_loss)
        print("{}[Epoch:{}/{}  Avg_train_loss:{:1.3f} Acc_train:{:3.2f}%]".format(("*"*30),i+1,epoch,avg_train_loss,train_accuracy))#loss.item()
        #print("{}[Epoch:{}/{}  Avg_train_loss:{:1.3f} ]".format(("*"*30),i+1,epoch,avg_train_loss))
        with torch.no_grad():
            net.eval()
            val_loss_reg = 0.0
            total_val = 0
            step_count = 0
            correct_val = 0
            
            
            for step, (batch_x,label_y) in enumerate(val_Loader):
                batch_x = torch.FloatTensor((batch_x.type(torch.FloatTensor)-127.5)/128.0)#(train_data-127.5)/128.0
                label_y = torch.LongTensor(label_y)
                h,w,c= batch_x.shape[1],batch_x.shape[2],batch_x.shape[3]
                input_shape = (-1,c,h,w)
                val = batch_x.view(input_shape).to(device)
                labels = label_y.to(device)
                features_512,outputs= net(val)
                val_loss = loss_func(outputs,labels)
    
                val_loss_reg +=val_loss.item()
                step_count += 1
                
                # ans = torch.max(feat,-1)[1].squeeze()
                # total_val += len(labels)
                # correct_val += (ans.cpu() == labels.cpu()).float().sum()
                ans=[]
                ans_cos,ans_phi=torch.max(outputs[0],1)[1].squeeze(),torch.max(outputs[1],1)[1].squeeze()
                if((ans_cos == ans_phi).all()):
                    ans = ans_cos
                total_val += len(labels)
                correct_val += (ans.cpu() == labels.cpu()).float().sum()
            
            val_accuracy = 100 * correct_val / float(total_val)
            validation_accuracy.append(val_accuracy)
            
            avg_val_loss = val_loss_reg/step_count
            validation_loss.append(avg_val_loss)
            print("{}[Epoch:{}/{}  Avg_val_loss:{:1.3f} Acc_val:{:3.2f}]".format(("*"*30),i+1,epoch,avg_val_loss,val_accuracy))
        
            if(val_accuracy>best_acc):
                    best_acc=val_accuracy
                    save_model(net, '{}/best_sphere20a_{}.pth'.format(save_file,i+1))
            torch.cuda.empty_cache()
            
        scheduler.step(avg_val_loss)
        lr = optimizer.param_groups[0]['lr']
        print("LR:{}".format(lr))
        #if(i!=0 and (i%15)==0):
        #    learn_rate *= 0.1
        
        

    save_model(net, '{}/sphere20a_{}.pth'.format(save_file,epoch))   
    loss_plt_show(epoch,training_loss,validation_loss,learn_rate,save_file)
    acc_plt_show(epoch,training_accuracy,validation_accuracy,learn_rate,save_file)
    
    #save_model(net, '{}_{}.pth'.format(sphere20a,epoch))
    #net2 = torch.load('net.pkl')
    #ans=torch.max(output,1)[1].squeeze().cuda()
    #final=F.softmax(output)
    
