import numpy as np
import matplotlib.pyplot as plt
import cv2
from matlab_cp2tform import get_similarity_transform_for_cv2
import os

def contrast_img(img1, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img1.shape

    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    #cv2.imshow('original_img', img)
    #cv2.imshow("contrast_img", dst)
    return dst


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #銳化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    #cv.imshow("custom_blur_demo", dst)
    return dst

def save_part_of(img_path,image,box):
    
    r_xmin,r_xmax,r_ymin,r_ymax=box[0]
    l_xmin,l_xmax,l_ymin,l_ymax=box[1]
    n_xmin,n_xmax,n_ymin,n_ymax=box[2]
    cv2.imwrite("{}_right.jpg".format(img_path), image[r_ymin:r_ymax,r_xmin:r_xmax])
    cv2.imwrite("{}_left.jpg".format(img_path), image[l_ymin:l_ymax,l_xmin:l_xmax])
    cv2.imwrite("{}_nose.jpg".format(img_path), image[n_ymin:n_ymax,n_xmin:n_xmax])

def plt_show(image):
    
    plt.imshow(image,cmap ='gray')
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()

if __name__ == '__main__':
    
    dog_img_path=[]
    dog_img_box=[]#nose,right_eye,left_eye
    
    dog_annotations=[]
    with open("save_Annotations.txt", "r") as f:
        for line in f.readlines():
                line = line.strip('\n') #去除換行
                reg=[]
                for word in line.split():
                    if(len(word.split(","))==1):
                        dog_img_path.append(word)
                    else:
                        #print(1)
                        #print(word.split(","))
                        reg.append([int(float(i)) for i in word.split(",")])
                dog_img_box.append(reg)
    
    shiba_path="Shiba"
    save_path="Shiba_result"
    if not os.path.isdir(os.path.join(save_path)):
            os.mkdir(os.path.join(save_path))
    for shiba_name in os.listdir(shiba_path):
        if not os.path.isdir(os.path.join(save_path,shiba_name)):
            os.mkdir(os.path.join(save_path,shiba_name))
        count=0
        for shiba_img in os.listdir(os.path.join(shiba_path,shiba_name)):
            _save=os.path.join(save_path,shiba_name,str(count))
            print(shiba_img)
            rpts=dog_img_box[dog_img_path.index(shiba_img)]
            img = cv2.imread(os.path.join(shiba_path,shiba_name,shiba_img))
            plt_show(img)
            ddd = custom_blur_demo(img)#銳化
            plt_show(ddd)
            ccc = contrast_img(ddd, 1.3, 3)#對比度與亮度
            plt_show(ccc)
            g_img = cv2.cvtColor(ccc, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(g_img)         #灰度图片直方图均衡化
            ret, thresh = cv2.threshold(eq, 10, 255,0)#二質化
            plt_show(thresh)
            #ddd=custom_blur_demo(thresh)
            #plt_show(ddd)
            blurred = cv2.GaussianBlur(thresh, (11, 11), 0)
            plt_show(blurred)
            ssave="{}_{}".format(_save,"Gauss")
            #cv2.imwrite("{}.jpg".format(ssave),blurred)
            #save_part_of(ssave,blurred,rpts)
            kernel = np.ones((3,3),np.uint8)  
            #dilation_kernel = np.ones((3,3),np.uint8)  
            erosion = cv2.erode(thresh,kernel,iterations = 3)
            #dilation = cv2.dilate(erosion,dilation_kernel,iterations = 2)
            plt_show(erosion)
            ssave="{}_{}".format(_save,"erode")
            #save_part_of(ssave,erosion,rpts)
            count+=1
            #plt_show(dilation)
    # for image_path,right_eye,left_eye,nose in dog_annotations:
    #     print(image_path,right_eye,left_eye,nose)
    #     img = cv2.imread(image_path)
    #     pts1=[right_eye,left_eye,nose]
    #     plt_show(img)
    #     dst = alignment(img,pts1)
        #plt_show(dst)
        
    
    #img = cv2.imread('nana0.jpg')
    #rows, cols, ch = img.shape
    #pts1 = np.float32([[70, 153],[150, 140],[115, 211]])
    #pts2 = np.float32([[30,51],[65,51],[48,71]])
    #M = cv2.getAffineTransform(pts1, pts2)
    
    #dst = alignment(img,pts1)
    #dst = cv2.warpAffine(img, M, (cols, rows),)
    #plt_show(dst)
    
    pass