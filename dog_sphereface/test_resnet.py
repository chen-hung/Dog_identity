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
import matplotlib.pyplot as plt

def get_dataset(BATCH_SIZE=10):
    
    #train_image = np.load('crop_dog/Shiba/train_image.npy')
    #train_label = np.load('crop_dog/Shiba/train_label.npy')
    #val_image = np.load('crop_dog/Shiba/val_image.npy')
    #val_label = np.load('crop_dog/Shiba/val_label.npy')
    class_name = np.load('0save_numpy/class_name.npy')
    #everybody = np.load('crop_dog/Shiba/everybody.npy')
    #class_everybody = np.load('crop_dog/Shiba/class_everybody.npy')
    
    '''
    torch_train_image = torch.from_numpy(train_image).type(torch.FloatTensor)
    torch_train_image = torch_train_image/255
    torch_train_label = torch.from_numpy(train_label).type(torch.LongTensor)
    
    torch_val_image = torch.from_numpy(val_image).type(torch.FloatTensor)
    torch_val_image = torch_val_image/255
    torch_val_label = torch.from_numpy(val_label).type(torch.LongTensor)
    
    train_tensor = torch.utils.data.TensorDataset(torch_train_image,torch_train_label)
    val_tensor = torch.utils.data.TensorDataset(torch_val_image,torch_val_label)
    
    train_Loader = DataLoader(dataset=train_tensor,batch_size=BATCH_SIZE,shuffle=True)
    val_Loader = DataLoader(dataset=val_tensor,batch_size=BATCH_SIZE,shuffle=True)
    '''
    #return train_Loader,val_Loader,class_name,everybody
    return class_name#,class_everybody

def plt_show(image):
    
    plt.imshow(image,cmap ='gray')
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()

def rotate(image, angle, center=None, scale=0.9):
    (h, w) = image.shape[:2] 
    if center is None: 
        center = (w // 2, h // 2) 

    M = cv2.getRotationMatrix2D(center, angle, scale) 

    rotated = cv2.warpAffine(image, M, (w, h)) 
    return rotated 

def get_image(path):
    
    input_image_path=path
    input_image = cv2.imread(input_image_path) #get_image(input_image_path)
    #constant= cv2.copyMakeBorder(input_image,200,200,250,250,cv2.BORDER_CONSTANT,value=(181,181,181))
    #constant=input_image
    #rotate_img = rotate(constant, 0,scale=1.5)
    rotate_img=input_image
    #input_image = path
    input_image = cv2.resize(rotate_img, (224,224))# w,h
    plt_show(input_image )
    torch_train_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    torch_train_image = torch_train_image/255
    
    batch = torch_train_image.unsqueeze(0)
    #print(torch_train_image.shape)
    #print(batch.shape)
    
    #plt_show(input_image)
    H,W,C=input_image.shape
    reshape_image = np.reshape(input_image,(1,H,W,C))
    transpose_image = np.transpose(reshape_image,(0,3,1,2))
    
    return batch

class_name=get_dataset()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_class=len(class_name)

net = models.resnet50(pretrained=False)
net.fc = torch.nn.Sequential(torch.nn.Linear(2048, 1000),
                                 torch.nn.Linear(1000, 500),
                                 torch.nn.Linear(500, 100),
                                 torch.nn.Linear(100, num_class),
                                 )
#net.fc = torch.nn.Linear(2048, num_class)

#net.load_state_dict(torch.load('save_resnet/net_100_0.08.pkl'))
net.load_state_dict(torch.load('0save_resnet/best.pkl'))
net = net.to(device)
net.eval()

if __name__ == "__main__":
    
    image_file=["5_Nana","7_Yutzu","0_Koro","8_Polly"]
    image_file1=["test4","test5","test8","test1"]
    image_file2=["test_nana","test_yutzu"]
    '''
    for i,images in enumerate(class_everybody):
            print("*"*60)
            print(class_name[i])
            #plt_show(images[0])
            img=get_image(images[0])
            h,w,c= img.shape[1],img.shape[2],img.shape[3]
            input_shape = (-1,c,h,w)
            test = Variable(img.view(input_shape)).to(device)
            final = net(test)
            #test=torch.FloatTensor(img).to(device)
            #final=net((test/255))
            #sigmod_final=F.sigmoid(final)
            #softmax_final=F.softmax(final)
            ans=torch.max(final,1)[1].squeeze()
            ans = ans.cpu().data.numpy()
            #print(class_name)
            #print(final.cpu().data.numpy())
            #print(sigmod_final)
            #print(softmax_final)
            print(class_name[ans])
            print(final[0][int(ans)])
            if(class_name[i]!=class_name[ans]):
                plt_show(cv2.imread(path))
    '''
    
    image_path="Shiba_test"
    all_count=0
    acc_count=0
    for f_class in os.listdir(image_path):
        #for name in os.listdir(os.path.join(image_path,f_class)):
            print("*"*60+"\n")
            print(f_class)
            path=os.path.join(image_path,f_class)
            img=get_image(path)
            h,w,c= img.shape[1],img.shape[2],img.shape[3]
            input_shape = (-1,c,h,w)
            test = Variable(img.view(input_shape)).to(device)
            final = net(test)
            #test=torch.FloatTensor(img).to(device)
            #final=net((test/255))
            sigmod_final=F.sigmoid(final)
            #softmax_final=F.softmax(final)
            ans=torch.max(final,1)[1].squeeze()
            ans = ans.cpu().data.numpy()
            #print(class_name)
            #print(final.cpu().data.numpy())
            #print(sigmod_final)
            #print(softmax_final)
            print(class_name[ans])
            print(sigmod_final[0][int(ans)])
            if(f_class[3]==class_name[ans][3]):
                acc_count+=1
                #plt_show(cv2.imread(path))
            plt_show(cv2.imread(path))
                
            all_count+=1
    print("正確率{}%".format(acc_count/all_count*100))
    
    '''
    for i in image_file:
        
        print("*"*60+"\n",i)
        path="./test/{}.jpg".format(i)
        img=get_image(path)
        h,w,c= img.shape[1],img.shape[2],img.shape[3]
        input_shape = (-1,c,h,w)
        test = Variable(img.view(input_shape)).to(device)
        final = net(test)
        #test=torch.FloatTensor(img).to(device)
        #final=net((test/255))
        #sigmod_final=F.sigmoid(final)
        #softmax_final=F.softmax(final)
        ans=torch.max(final,1)[1].squeeze()
        ans = ans.cpu().data.numpy()
        print(class_name)
        print(final.cpu().data.numpy())
        #print(sigmod_final)
        #print(softmax_final)
        print(class_name[ans])
        print(final[0][int(ans)])
    '''
    pass