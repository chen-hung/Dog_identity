import torch
import torchvision
import argparse
import cv2
import numpy as np
import sys
sys.path.append('./')
import coco_names
import random
from PIL import Image
from torchvision import transforms
import os

def cv_imread(filePath):
    image = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),cv2.IMREAD_COLOR)
    return image

def RCNN_dog(model,img_path,save_name,score=0.8):
    names = coco_names.names
    input=[]
    save_file="images"#"result"
    if not os.path.isdir(save_file):
        os.mkdir(save_file)
    
    image_path=img_path
    src_img = cv_imread(image_path)#(args.image_path)
    img = cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().cuda()
    input.append(img_tensor)
    out = model(input)
    
    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']
    
    crop_img = src_img.copy()
    class_count = 0
    
    yes_no_dog =0
    
    for idx in range(boxes.shape[0]):
        if scores[idx] >= score:
            x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
            name = names.get(str(labels[idx].item()))
            # cv2.rectangle(img,(x1,y1),(x2,y2),colors[labels[idx].item()],thickness=2)
            if(name =="dog"):
                cv2.imwrite('./{}/{}_result_dog{}.jpg'.format(save_file,save_name,class_count),crop_img[y1:y2,x1:x2])
                class_count+=1
            
                cv2.rectangle(src_img,(x1,y1),(x2,y2),random_color(),thickness=5)
                yes_no_dog=1
                #cv2.putText(src_img, text=name, org=(x1, y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                #fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

    #cv2.imwrite('./{}/result.jpg'.format(save_file),src_img)
    
    if(yes_no_dog!=0):
        return 1,src_img
    else:
        return 0,src_img

def random_color():
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)

    return (b,g,r)


    

if __name__ == "__main__":
    
    img_path="./測試/nana_1.jpg"
    RCNN_dog(img_path,score=0.8)
    
