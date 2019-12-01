import os
import cv2
data_set = "/media/gytang/Seagate Expansion Drive/206服务器/CycleGAN-master4/datasets/align/train"
for img in os.listdir(data_set):
    ig = cv2.imread(os.path.join(data_set,img))
    clear = ig[:,:int(ig.shape[1]/2),:]
    haze = ig[:,int(ig.shape[1]/2):,:]
    clear = cv2.resize(clear,(128,128))
    haze = cv2.resize(haze, (128, 128))
    cv2.imwrite(os.path.join("./align/clear/",img),clear)
    cv2.imwrite(os.path.join("./align/haze/", img), haze)

