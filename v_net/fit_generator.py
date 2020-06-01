import numpy as np
import cv2 as cv
import os

data_train_path = "../../Vnet_tf/V_Net_data/train"
data_label_path = "../../Vnet_tf/V_Net_data/label"


def get_path_list(data_train_path, data_label_path):
    dirs = os.listdir(data_train_path)
    dirs.sort(key=lambda x: int(x))
    count = 0
    for dir in dirs:
        dir_path = os.path.join(data_train_path, dir)
        count += len(os.listdir(dir_path))
    print("共有{}组训练数据".format(count))

    train_path_list = []
    label_path_list = []
    for dir in dirs:
        train_dir_path = os.path.join(data_train_path, dir)
        label_dir_path = os.path.join(data_label_path, dir)
        trains = os.listdir(train_dir_path)
        labels = os.listdir(label_dir_path)
        trains.sort(key=lambda x: int(x.split(".")[0]))
        labels.sort(key=lambda x: int(x.split(".")[0]))
        for name in trains:
            train_path = os.path.join(train_dir_path, name)
            label_path = os.path.join(label_dir_path, name)

            train_path_list.append(train_path)
            label_path_list.append(label_path)

    return train_path_list, label_path_list, count


def get_train_img(paths, img_d, img_rows, img_cols):
    """
    参数：
        paths：要读取的图片路径列表
        img_rows:图片行
        img_cols:图片列
        color_type:图片颜色通道
    返回:
        imgs: 图片数组
    """
    # Load as grayscale
    datas = []
    for path in paths:
        data = np.load(path)
        # Reduce size
        resized = np.reshape(data, (img_d, img_rows, img_cols, 1))
        resized = resized.astype('float32')
        resized /= 255
        # 均值
        # 注意：这里取均值时，要考虑是输入一批train还是单个train
        # 一批train需要设置axis=0，这样就是对每一张图像求均值
        # 单个train，就不需要设置，就会直接对图像求均值
#        mean = resized.mean(axis=0)
        # 标准差
        # std = np.std(resized, ddof=1)
        # 标准化
#        resized -= mean
        # resized /= std
        datas.append(resized)
    datas = np.array(datas)
    return datas


def get_label_img(paths, img_rows, img_cols):
    """
    参数：
        paths：要读取的图片路径列表
        img_rows:图片行
        img_cols:图片列
        color_type:图片颜色通道
    返回:
        imgs: 图片数组
    """
    # Load as grayscale
    datas = []
    for path in paths:
        data = np.load(path)
        # Reduce size
        resized = np.reshape(data, (1, img_cols, img_rows, 1))
        resized = resized.astype('float32')
        resized /= 255
        datas.append(resized)
    datas = np.array(datas)
    return datas


def get_train_batch(train, label, batch_size, img_d, img_w, img_h):
    """
    参数：
        X_train：所有图片路径列表
        y_train: 所有图片对应的标签列表
        batch_size:批次
        img_w:图片宽
        img_h:图片高
        color_type:图片类型
        is_argumentation:是否需要数据增强
    返回:
        一个generator，x: 获取的批次图片 y: 获取的图片对应的标签
    """
    while 1:
        for i in range(0, len(train), batch_size):
            x = get_train_img(train[i:i+batch_size], img_d, img_w, img_h)
            y = get_label_img(label[i:i+batch_size], img_w, img_h)
            # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
            yield(np.array(x), np.array(y))


if __name__ == "__main__":
    train_path_list, label_path_list, count = get_path_list(data_train_path, data_label_path)
    print(train_path_list)

