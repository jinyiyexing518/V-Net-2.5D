"""
本程序将服务器的dir---npy文件拆解成dir---png文件
注意：这里标签是纯肝脏数据，肿瘤作为背景
"""
import os
import cv2 as cv
import numpy as np


train_png_path = "./data_dir_png/train"
label_png_path = "./data_dir_png/label"
train_npy_path = "./data_dir_npy_pre_mean/train"
label_npy_path = "./data_dir_npy_pre_mean/label"
if not os.path.isdir(train_png_path):
    os.makedirs(train_png_path)
if not os.path.isdir(label_png_path):
    os.makedirs(label_png_path)

train_npys = os.listdir(train_npy_path)
label_npys = os.listdir(label_npy_path)
train_npys.sort(key=lambda x: int(x.split(".")[0]))
label_npys.sort(key=lambda x: int(x.split(".")[0]))

j = 0
for npy in train_npys:
    npy_path1 = os.path.join(train_npy_path, npy)
    npy_path2 = os.path.join(label_npy_path, npy)
    train_npy = np.load(npy_path1)
    label_npy = np.load(npy_path2)
    for i in range(len(train_npy)):
        train_img = train_npy[i]
        label_img = label_npy[i]
        train_img = np.reshape(train_img, (400, 400))
        label_img = np.reshape(label_img, (400, 400))

        # cv.imshow("train", train_img)
        # cv.imshow("label", label_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        train_save_dir_path = os.path.join(train_png_path, str(j))
        label_save_dir_path = os.path.join(label_png_path, str(j))
        if not os.path.isdir(train_save_dir_path):
            os.makedirs(train_save_dir_path)
        if not os.path.isdir(label_save_dir_path):
            os.makedirs(label_save_dir_path)
        cv.imwrite(train_save_dir_path + "/" + str(i) + ".png", train_img)
        cv.imwrite(label_save_dir_path + "/" + str(i) + ".png", label_img)
    j += 1
    print("完成第{}个dir".format(j))






