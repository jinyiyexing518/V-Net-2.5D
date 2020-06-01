"""
本程序制作vnet数据集，同一dir中的11张切片作为一组输入
服务器存储空间有限，一个dir一个dir进行制作，然后训练
"""
import os
import cv2 as cv
import numpy as np


# 制作哪一个dir的数据
num = 66
for i in range(num):
    dir_num = i
    image_size = 400
    depth = 3

    train_png_path = "./data_dir_png/train"
    label_png_path = "./data_dir_png/label"
    train_64_input_path = "./vnet_" + str(depth) + "_1_input/train"
    label_64_input_path = "./vnet_" + str(depth) + "_1_input/label"
    if not os.path.isdir(train_64_input_path):
        os.makedirs(train_64_input_path)
    if not os.path.isdir(label_64_input_path):
        os.makedirs(label_64_input_path)

    train_dirs = os.listdir(train_png_path)
    label_dirs = os.listdir(label_png_path)
    train_dirs.sort(key=lambda x: int(x))
    label_dirs.sort(key=lambda x: int(x))
    # 选择需要制作数据的dir
    dir_name = train_dirs[dir_num]
    # train和label的dir地址
    train_dir_path = os.path.join(train_png_path, dir_name)
    label_dir_path = os.path.join(label_png_path, dir_name)

    train_pngs = os.listdir(train_dir_path)
    train_pngs.sort(key=lambda x: int(x.split(".")[0]))
    label_pngs = os.listdir(label_dir_path)
    label_pngs.sort(key=lambda x: int(x.split(".")[0]))

    for i in range(len(train_pngs)):
        train_npy = np.ndarray((depth, image_size, image_size, 1), dtype=np.uint8)
        label_npy = np.ndarray((image_size, image_size, 1), dtype=np.uint8)
        if (i + depth-1) < len(train_pngs):
            label_img_path = os.path.join(label_dir_path, label_pngs[i+1])
            label_img = cv.imread(label_img_path, 0)

            # cv.imshow("label", label_img)

            label_img = np.reshape(label_img, (image_size, image_size, 1))
            label_npy = label_img
            for j in range(depth):
                index = i + j
                train_img_path = os.path.join(train_dir_path, train_pngs[index])
                train_img = cv.imread(train_img_path, 0)

                # cv.imshow("train", train_img)

                train_img = np.reshape(train_img, (image_size, image_size, 1))
                train_npy[j] = train_img

                # cv.waitKey(0)
                # cv.destroyAllWindows()

            train_input_path = os.path.join(train_64_input_path, str(dir_num))
            label_input_path = os.path.join(label_64_input_path, str(dir_num))
            if not os.path.isdir(train_input_path):
                os.makedirs(train_input_path)
            if not os.path.isdir(label_input_path):
                os.makedirs(label_input_path)
            print(train_npy.shape)
            print(label_npy.shape)
            np.save(train_input_path + "/" + str(i) + ".npy", train_npy)
            np.save(label_input_path + "/" + str(i) + ".npy", label_npy)














