import numpy as np 
import cv2 as cv
import nibabel as nib
import os
from PIL import Image
import imageio


def transform_ctdata(image, windowWidth, windowCenter, normal=False):
    """
    注意，这个函数的self.image一定得是float类型的，否则就无效！
    return: trucated image according to window center and window width
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (image - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


# nii文件存放路径
train_path = "D:/pycharm_project/graduate_design_next_semester/dataset/data_nii/train"
label_path = "D:/pycharm_project/graduate_design_next_semester/dataset/data_nii/label"
# slice存放路径
train_save_path = './data_raw_slice_tumour/train'
label_save_path = './data_raw_slice_tumour/label'
if not os.path.isdir(train_save_path):
    os.makedirs(train_save_path)
if not os.path.isdir(label_save_path):
    os.makedirs(label_save_path)


# 准备导入训练图像以及标签，并对图像和标签进行排序
train_images = os.listdir(train_path)
train_images.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
label_images = os.listdir(label_path)
label_images.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))


def create_train_label_slice():
    print('-' * 30)
    print("同时分解volume与segmentation文件")
    for i in range(len(train_images)):
        train_image = nib.load(train_path + '/' + train_images[i])
        label_image = nib.load(label_path + '/' + label_images[i])
        # 获取每一个nii文件的行、列、切片数
        height, width, slice = train_image.shape
        print("第" + str(i) + "个dir", "(", height, width, slice, ")")
        # 保存切片的小子文件夹序号，0，1，2等
        slice_save_path = train_images[i].split('-')[1].split('.')[0]

        train_slice_path = train_save_path + '/' + slice_save_path
        label_slice_path = label_save_path + '/' + slice_save_path

        # if not os.path.isdir(train_slice_path):
        #     os.makedirs(train_slice_path)
        # if not os.path.isdir(label_slice_path):
        #     os.makedirs(label_slice_path)

        img_fdata = label_image.get_fdata()
        for j in range(slice):
            train_img = train_image.dataobj[:, :, j]
            label_img = img_fdata[:, :, j]

            label_img[label_img == 1] = 0
            label_img[label_img == 2] = 1

            white_pixel = label_img == 1
            white_pixel_num = len(label_img[white_pixel])

            # # 判断是否为全黑的标签，这样没有意义，剔除
            # if label_img.max() != 0:

            # 肿瘤标签像素点数量应该大于50，才算作有效数据
            if white_pixel_num >= 50:
                set_slice = np.array(train_img).copy()
                set_slice = set_slice.astype("float32")
                # 训练用的窗位窗宽
                # set_slice = transform_ctdata(set_slice, 350, 25)
                # 知乎上参考，肝脏是40~60
                set_slice = transform_ctdata(set_slice, 200, 30)

                # set_slice = set_slice.astype("float32")
                # mean = set_slice.mean()
                # std = np.std(set_slice)
                # set_slice -= mean
                # set_slice /= std
                # set_slice = (set_slice - set_slice.min()) / (set_slice.max() - set_slice.min())
                # set_slice *= 255
                # # set_slice = transform_ctdata(set_slice, 250, 125)
                # set_slice = set_slice.astype("uint8")

                # 中值滤波，去除椒盐噪声
                set_slice = cv.medianBlur(set_slice, 3)

                if not os.path.isdir(train_slice_path):
                    os.makedirs(train_slice_path)
                if not os.path.isdir(label_slice_path):
                    os.makedirs(label_slice_path)

                # 加入直方图均衡处理
                # set_slice = cv.equalizeHist(set_slice)
                cv.imwrite(train_slice_path + '/' + str(j) + '.png', set_slice)
                label_img = Image.fromarray(np.uint8(label_img * 255))
                imageio.imwrite(label_slice_path + '/' + str(j) + '.png', label_img)
            else:
                pass
    print("Generating train data set done!")


if __name__ == "__main__":
    # generate_liver_data()

    create_train_label_slice()

    # nii_path = "../../dataset/data_nii/train/volume-0.nii"
    # nii_file = nib.load(nii_path)
    #
    # # 获取每一个nii文件的行、列、切片数
    # height, width, depth = nii_file.shape
    #
    # for i in range(depth):
    #     slice = nii_file.dataobj[:, :, i]
    #
    #     plt.figure(1)
    #     plt.hist(slice.ravel(), 256, [1, 256])
    #
    #     set_slice = np.array(slice).copy()
    #     set_slice = transform_ctdata(set_slice, 450, 25)
    #
    #     set_slice = cv.equalizeHist(set_slice)
    #
    #     print(set_slice.dtype, set_slice.max())
    #
    #     plt.figure(2)
    #     plt.hist(set_slice.ravel(), 256, [1, 256])
    #
    #     plt.figure(3)
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(slice, cmap="gray")
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(set_slice, cmap="gray")
    #
    #     plt.show()



