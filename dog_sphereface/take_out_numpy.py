import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import gc

def RGB_equalizeHist(image):
    
    (b, g, r) = cv2.split(image)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    frameH = cv2.merge((bH, gH, rH))
    
    return frameH

def plt_show(image):
    
    plt.imshow(image,cmap ='gray')
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig("test.jpg")
    plt.show()
    
def rotate(image, angle, center=None, scale=0.9):
    (h, w) = image.shape[:2] 
    if center is None: 
        center = (w // 2, h // 2) 

    M = cv2.getRotationMatrix2D(center, angle, scale) 

    rotated = cv2.warpAffine(image, M, (w, h)) 
    return rotated 

if __name__ == "__main__":
    
    train_image_label=[]
    val_image_label=[]
    
    image_label=[]
    
    class_name=np.array([]) 
    
    img_path="part_of_dog_face_part2"
    resize = (224,224)
    
    
    
    for dog_name in os.listdir(img_path):
             class_name=np.append(class_name,dog_name)
             label = int(np.where(class_name==dog_name)[0])
             for dog_image in os.listdir(os.path.join(img_path,dog_name)):
                 print(dog_image)
                 image = cv2.imread(os.path.join(img_path,dog_name,dog_image))
                 plt_show(image)
                 if(dog_image[-6:]in["_3.jpg","_2.jpg"]):
                     val_image_label.append([image,label])
                 else:
                     train_image_label.append([image,label])
                     image_label.append([image,label])
                     flip_image = cv2.flip(image, 1)
                     plt_show(flip_image)
                     train_image_label.append([flip_image,label])
                     rgb_image = RGB_equalizeHist(image)
                     plt_show(rgb_image)
                     train_image_label.append([rgb_image,label])

    
    train_image_label=np.array(train_image_label)
    val_image_label=np.array(val_image_label)
    
    print("Total:{}".format(len(image_label)))
    print("Train:{}".format(len(train_image_label)))
    print("Val:{}".format(len(val_image_label)))
    # # print("Test:{}".format(len(test_image)))
    
    
    path="./save_numpy"
    if not os.path.isdir(path):
        os.mkdir(path)
    
    np.save("{}/train_image_label".format(path),train_image_label)
    np.save("{}/class_name".format(path),class_name)
    np.save("{}/val_image_label".format(path),val_image_label)
    np.save("{}/image_label".format(path),image_label)
        
    pass
