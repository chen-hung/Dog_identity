import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import dlib
from imutils import face_utils
from matlab_cp2tform import get_similarity_transform_for_cv2


def alignment_test(src_img,src_pts):
    of = 0
    ref_pts = [ [30.2946+of, 51.6963+of],[65.5318+of, 51.5014+of],[48.0252+of, 71.7366+of]]
    crop_size =(96+of*2, 112+of*2)
    xx1,xx2,yy1,yy2=30,65,51,92
    #print(src_img.shape[1],src_img.shape[0])
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)
    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    plt_show(face_img)
    rx,ry=12,14
    crop_face_img=face_img[yy1-ry:yy2+ry,xx1-rx:xx2+rx]
    resize_face_img=cv2.resize(crop_face_img, (96, 112))
    return resize_face_img

def alignment(src_img,src_pts):#校正人臉
    of = 0
    #ref_pts = [ [30.2946+of, 51.6963+of],[65.5318+of, 51.5014+of],[48.0252+of, 71.7366+of] ]
    ref_pts = [ [86+of,140+of],[172+of,140+of],[129+of,170+of]]#[29,80][159,80][96,157]
    crop_size =(304,320)#(96+of*2, 112+of*2)#(144+of*2,160+of*2)
    
    # ref_pts = [ [86+of,80+of],[300+of,80+of],[129+of,120+of]]
    # crop_size =(304,320)#src_img.shape[:2]#(160,144)
    
    xx1,xx2,yy1,yy2=76,182,105,200#20,75,46,76
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)
    tfm = get_similarity_transform_for_cv2(s, r)
    
    M=cv2.getAffineTransform(s,r)
    
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    
    #face_img = cv2.warpAffine(src_img, M, crop_size)
    plt_show(face_img)
    
    
    rx,ry=0,0
    crop_face_img=face_img[yy1-ry:yy2+ry,xx1-rx:xx2+rx]
    
    plt_show(crop_face_img)
    
    #crop_face_img=face_img[yy1-ry:yy2+ry,xx1-rx:xx2+rx]
    resize=(160,144)#(96, 112)
    #plt_show(face_img)
    #plt_show(crop_face_img)
    resize_face_img=cv2.resize(crop_face_img,resize)#crop_face_img
    #cv2.imwrite('output.jpg', resize_face_img)
    return resize_face_img


def detect_dog_face(image):
            img_result = image.copy()
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt_show(image)
            
            img = cv2.resize(img, dsize=None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            dets = detector(img, upsample_num_times=1)
            
            right_eye,left_eye,nose,crop_face= (),(),(),()
            face_x1,face_x2,face_y1,face_y2 = 0,0,0,0
            for i, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
            
                x1, y1 = int(d.rect.left() / SCALE_FACTOR), int(d.rect.top() / SCALE_FACTOR)
                x2, y2 = int(d.rect.right() / SCALE_FACTOR), int(d.rect.bottom() / SCALE_FACTOR)
                
                face_x1,face_x2,face_y1,face_y2 = x1,x2,y1,y2
                
                crop_face = image[y1:y2,x1:x2]
                
                cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(0,0,255), lineType=cv2.LINE_AA)
                
                shape = predictor(img, d.rect)
                shape = face_utils.shape_to_np(shape)
                
                for i, p in enumerate(shape):
                  local=tuple((p / SCALE_FACTOR).astype(int))
                  cv2.circle(img_result, center=local, radius=8, color=(255,255,0), thickness=-1, lineType=cv2.LINE_AA)
                  print(i,local)
                  cv2.putText(img_result, str(i), tuple((p / SCALE_FACTOR).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                  if i==5:
                      right_eye=[local[0],local[1]]
                  elif i==2:
                      left_eye=[local[0],local[1]]
                  elif i==3:
                      nose=[local[0],local[1]]
            
            plt_show(img_result)
            print("*"*40)
            
            return ([right_eye,left_eye,nose],[face_x1,face_x2,face_y1,face_y2],crop_face)
        
def plt_show(image):
    
    if(len(image.shape)==3):
        image = image[:,:,::-1]
    elif(len(image.shape)==2):
        image = image
    plt.imshow(image,cmap ='gray')
    plt.xticks([])
    plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()
    

SCALE_FACTOR = 0.2

    
detector = dlib.cnn_face_detection_model_v1('./data/dogHeadDetector.dat')
predictor = dlib.shape_predictor('./data/landmarkDetector.dat')

if __name__ == '__main__':
    

    
    path = r'./train_dog_dataset'
    save_path = "./part_of_dog_face_part2"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    image = cv2.imread("test/Shinjiro_30.jpg")#("Mogu_01.jpg")
    local,bound,face= detect_dog_face(image)
    part_dog_face = alignment(image,local)
    #part_dog_face = alignment_test(face,local)
   
    
    #part_dog_face = alignment(image,local)
    
    # for dog_name in os.listdir(path):
    #     print("{}".format(dog_name))
    #     if not os.path.isdir(os.path.join(save_path,dog_name)):
    #         os.mkdir(os.path.join(save_path,dog_name))
        
    #     for i,dog_image in enumerate(os.listdir(os.path.join(path,dog_name))):
    #         print(i,dog_image)
    #         image_path = os.path.join(path,dog_name,dog_image)
    #         image = cv2.imread(image_path)
    #         #image = cv2.resize(image,(400,400))
    #         local = detect_dog_face(image)
    #         if(len(local[0])!=0):
    #             part_dog_face = alignment(image,local)
    #             plt_show(part_dog_face)
    #             #cv2.imwrite(os.path.join(save_path,dog_name,dog_image),part_dog_face)
            



