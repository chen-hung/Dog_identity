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
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
#from sphereface import AngleSoftmax
from module.sphere_face import *
from module.loss import *
from scipy.spatial.distance import cosine
from crop_part_of import detect_dog_face,alignment
from detect import RCNN_dog
import torchvision

torch.backends.cudnn.bencmark = True

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def detect_dog_breed(net,image_path,device,class_name):
    

    crop_size=(224,224)
    img = cv2.imread(image_path)
    H,W,C = img.shape
    re_img = cv2.resize(img, crop_size)
    im_tfs = transforms.Compose([
        transforms.ToTensor(),
        #tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = im_tfs(re_img)
    batch = img_tensor.unsqueeze(0)
    test = Variable(batch).to(device)
    outputs = net(test)
    print(outputs)
    label_pred = outputs.max(1)[1].squeeze().cpu().data.numpy()
    #plt_show(img)
    print(class_name[label_pred])
    return class_name[label_pred]

def resizeimg(img,min_side=300):
    size = img.shape
    h, w = size[0], size[1]
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    return resize_img

def plt_show(image):
    
    plt.imshow(image)
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()

def Noot_dog(save_name):
    
    img = np.zeros((25, 280, 3), np.uint8)
    img.fill(255)
    pilimg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("mingliu.ttc", 20, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
    draw.text((0, 0), "沒偵測到犬隻", (0, 0, 0), font=font)
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    #cv2.putText(img, "品種:{}".format(breed), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite('./images/{}_breed0.jpg'.format(save_name), cv2charimg)
    #cv2.imwrite('./images/{}_sphereface.jpg'.format(save_name), cv2charimg)

# def Noot(save_name):
    
#     img = np.zeros((25, 280, 3), np.uint8)
#     img.fill(255)
#     pilimg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(pilimg)
#     font = ImageFont.truetype("mingliu.ttc", 20, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
#     draw.text((0, 0), "非柴犬，無資料", (0, 0, 0), font=font)
#     cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
#     #cv2.putText(img, "品種:{}".format(breed), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
#     cv2.imwrite('./images/{}_sphereface.jpg'.format(save_name), cv2charimg)
    

def last_detect(image_path,save_name):
    
    part_of = get_part_of(image_path)
    batch = np_get_image(part_of)
    features_512,ans_cos = get_features(batch)
    sss = detect(features_512,ans_cos,every_body)#detect(features_512,ans_cos,every_body)
    print("辨識結果:{}，相似度:{:3.2f}%".format(class_name[ans_cos],sss))
    img = np.zeros((25, 280, 3), np.uint8)
    img.fill(255)
    pilimg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("mingliu.ttc", 20, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
    draw.text((0, 0), "辨識結果:{}，相似度:{:3.2f}%".format(class_name[ans_cos],sss), (0, 0, 0), font=font)
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    #cv2.putText(img, "品種:{}".format(breed), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite('./images/{}_sphereface.jpg'.format(save_name), cv2charimg)
    #plt_show(cv2charimg)
    

def get_part_of(image_path):
            image = cv2.imread(image_path)
            local,bound,face = detect_dog_face(image)
            if(len(local[0])!=0):
                part_dog_face = alignment(image,local)
                #plt_show(part_dog_face)
            return part_dog_face

def np_get_image(input_image):

    
    torch_train_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    torch_train_image = (torch_train_image-127.5)/128.0
    batch = torch_train_image.unsqueeze(0)
    
    return  batch
    

def get_features(batch):
    
    h,w,c= batch.shape[1],batch.shape[2],batch.shape[3]
    input_shape = (-1,c,h,w)
    val = Variable(batch.view(input_shape)).to(device)
    features_512,outputs= net(val)
    ans_cos,ans_phi=torch.max(outputs[0],1)[1].squeeze(),torch.max(outputs[1],1)[1].squeeze()
    #print(class_name[ans_cos])
    
    return features_512,ans_cos

def detect(features,ans,every_body):
    
    sss=0
    for image in every_body[ans]:
        batch = np_get_image(image)
        features_512,_ = get_features(batch)
        score  = 1 - cosine(features.data.cpu().numpy(),features_512.data.cpu().numpy())
        sss+=score
        
    return (sss/len(every_body[ans])*100.0)
    pass

def get_image(path):
    
    input_image_path=path
    input_image = cv2.imread(input_image_path) #get_image(input_image_path)
    #plt_show(input_image)
    
    torch_train_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    torch_train_image = (torch_train_image-127.5)/128.0
    batch = torch_train_image.unsqueeze(0)
    
    return batch

def get_dataset(BATCH_SIZE):
    
    path="save_numpy"
    
    train_image_label= np.load(os.path.join(path,'train_image_label.npy'),allow_pickle=True)
    train_image_label = train_image_label.tolist()

    val_image_label= np.load(os.path.join(path,'val_image_label.npy'),allow_pickle=True)
    val_image_label = val_image_label.tolist()
    
    class_name= np.load(os.path.join(path,'class_name.npy'),allow_pickle=True)
    
    train_Loader = DataLoader(dataset=train_image_label,batch_size=BATCH_SIZE,shuffle=True)
    val_Loader = DataLoader(dataset=val_image_label,batch_size=4,shuffle=True)
    
    image_label= np.load(os.path.join(path,'image_label.npy'),allow_pickle=True)
    
    return train_Loader,val_Loader,image_label,class_name

def detect_dog(image_path,save_name):
    
    haha,result = RCNN_dog(model,image_path,save_name)
    
    if(haha==0):
        Noot_dog(save_name)
        #Noot(save_name)
    else:
        re_img = resizeimg(result)
        #plt_show(re_img)
    
        enable = detect_breed(save_name)
        #detect_sphereface()
        if(enable):
            detect_dog_sphereface("./images/{}_result_dog0.jpg".format(save_name),save_name)
        else:
            print("No")
            #Noot(save_name)

        
        
def detect_breed(save_name):
    
    image_path="./images/{}_result_dog0.jpg".format(save_name)
    breed = detect_dog_breed(breed_net,image_path,device,breed_class_name)
    print(breed)
    img = np.zeros((25, 120, 3), np.uint8)
    img.fill(255)
    pilimg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("mingliu.ttc", 20, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
    draw.text((0, 0), "品種:{}".format(breed), (0, 0, 0), font=font)
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    #cv2.putText(img, "品種:{}".format(breed), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite('./images/{}_breed0.jpg'.format(save_name), cv2charimg)
    
    if(breed!="柴犬"):
        return 0
    else:
        return 1
    #plt_show(cv2charimg)

def detect_dog_sphereface(image_path,save_name):
    
        
    #image_path ="./Mogu_01.jpg"#= os.path.join(path,dog_name,dog_image)
    last_detect(image_path,save_name)
    print("原圖:{}".format(image_path))
    print("="*100)
    
def get_labels(root):
    f = open(root, 'r') 
    labels = []
    for line in f :
            line = line.rstrip()
            #print(line)
            labels.append(line)
    return labels

batch_size=48
train_Loader,val_Loader,image_label,class_name=get_dataset(batch_size)
num_class=len(class_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SphereFace20(num_classes=num_class, feat_dim=512).to(device)

net.load_state_dict(torch.load('save_sphere_part2/best_sphere20.pth'))
net.eval()

every_body=[]

for i in range(len(class_name)):
    reg=[]
    for image,label in image_label:
        if(label==i):
            reg.append(image)
    every_body.append(reg)

print("Creating model")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.cuda()
model.eval()

root_label = "./save_txt/Labels.txt"
breed_class_name = get_labels(root_label)
breed_num_class = len(breed_class_name)

breed_net = models.resnet50(pretrained=False)

breed_net.fc = torch.nn.Sequential(torch.nn.Linear(2048, 1000),
                             torch.nn.Linear(1000, 500),
                             torch.nn.Linear(500, 100),
                             torch.nn.Linear(100, breed_num_class),)

breed_net = breed_net.to(device)
breed_net.eval()
breed_net.load_state_dict(torch.load("save_txt/save_net_best_37.150_0.0484375.pkl"))
#(torch.load('save_txt/save_net_best_150.150_0.0484375.pkl'))



if __name__ == "__main__":
    
    # image_path="./Mogu_01.jpg"
    # detect_dog_sphereface(image_path)
    save_name="123"
    
    image_path="./987654.jpg"
    detect_dog(image_path,save_name)
    
    # threshold_val=50
    
    # batch_size=48
    # train_Loader,val_Loader,image_label,class_name=get_dataset(batch_size)
    # num_class=len(class_name)
    
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = SphereFace20(num_classes=num_class, feat_dim=512).to(device)
    # #loss_func = AngularSoftmaxWithLoss()
    # net.load_state_dict(torch.load('save_sphere_part2/best_sphere20.pth'))
    # net.eval()
    
    # every_body=[]
    
    # for i in range(len(class_name)):
    #     reg=[]
    #     for image,label in image_label:
    #         if(label==i):
    #             reg.append(image)
    #     every_body.append(reg)
        
    # #image_path="./Aco_2.jpg"
    # path = r'./test_dog_dataset'
    # for dog_name in os.listdir(path):
    #     for i,dog_image in enumerate(os.listdir(os.path.join(path,dog_name))):
    #         print("="*100)
    #         print(i,dog_image)
    #         image_path = os.path.join(path,dog_name,dog_image)
    #         last_detect(image_path)
    #         print("原圖:{}".format(dog_image))
    #         print("="*100)
    
    #**************************************************************************************
    # image_path ="./Mogu_01.jpg"#= os.path.join(path,dog_name,dog_image)
    # last_detect(image_path)
    # print("原圖:{}".format(image_path))
    # print("="*100)
    #**************************************************************************************
    # if(1):
       
    #     train_loss_reg = 0.0
    #     total_train = 0
    #     step_count = 0
    #     correct_train = 0
    #     #scatter_x , scatter_y,scatter_color =[],[],[]
    #     for step, (batch_x,label_y) in enumerate(train_Loader):
    #         batch_x = torch.FloatTensor((batch_x.type(torch.FloatTensor)-127.5)/128.0)#(train_data-127.5)/128.0
    #         label_y = torch.LongTensor(label_y)
    #         h,w,c= batch_x.shape[1],batch_x.shape[2],batch_x.shape[3]
    #         input_shape = (-1,c,h,w)
    #         train = batch_x.view(input_shape).to(device)#,volatile=True)
    #         labels = label_y.to(device)#,volatile=True)
    #         features_512,outputs= net(train)
    #         step_count += 1
    #         ans=[]
    #         ans_cos,ans_phi=torch.max(outputs[0],1)[1].squeeze(),torch.max(outputs[1],1)[1].squeeze()
    #         if((ans_cos == ans_phi).all()):
    #             ans = ans_cos
    #         total_train += len(labels)
    #         correct_train += (ans.cpu() == labels.cpu()).float().sum()
    #     train_accuracy = 100 * correct_train / float(total_train)
    #     print("Acc_train:{:3.2f}%".format(train_accuracy))
    #     val_loss_reg = 0.0
    #     total_val = 0
    #     step_count = 0
    #     correct_val = 0

    #     for step, (batch_x,label_y) in enumerate(val_Loader):
    #             batch_x = torch.FloatTensor((batch_x.type(torch.FloatTensor)-127.5)/128.0)#(train_data-127.5)/128.0
    #             label_y = torch.LongTensor(label_y)
    #             h,w,c= batch_x.shape[1],batch_x.shape[2],batch_x.shape[3]
    #             input_shape = (-1,c,h,w)
    #             val = Variable(batch_x.view(input_shape)).to(device)
    #             labels = Variable(label_y).to(device)
    #             features_512,outputs= net(val)
    #             step_count += 1
                
    #             ans=[]
    #             ans_cos,ans_phi=torch.max(outputs[0],1)[1].squeeze(),torch.max(outputs[1],1)[1].squeeze()
    #             if((ans_cos == ans_phi).all()):
    #                 ans = ans_cos
    #             total_val += len(labels)
    #             correct_val += (ans.cpu() == labels.cpu()).float().sum()
                
    #     val_accuracy = 100 * correct_val / float(total_val)
    #     avg_val_loss = val_loss_reg/step_count
    #     print("Acc_val:{:3.2f}%".format(val_accuracy))
    
    
    # everybody_features = []
    # for i in range(len(class_everybody)):
    #     reg=[]
    #     print(i)
    #     for input_image in class_everybody[i]:
    #          #H,W,C=input_image.shape
    #          #reshape_image = np.reshape(input_image,(1,H,W,C))
    #          #transpose_image = np.transpose(reshape_image,(0,3,1,2))
    #          #test=torch.FloatTensor(transpose_image).to(device)
    #          img=np_get_image(input_image)
    #          h,w,c= img.shape[1],img.shape[2],img.shape[3]
    #          input_shape = (-1,c,h,w)
    #          test = Variable(img.view(input_shape)).to(device)
    #          outputs,features_512 = net(test)
    #          features_512=features_512.data.cpu().numpy()
    #          reg.append([features_512])
    #     torch.cuda.empty_cache()
    #     everybody_features.append(reg)
    # everybody_features=np.array(everybody_features)
    
    

    
    
    
    pass
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    