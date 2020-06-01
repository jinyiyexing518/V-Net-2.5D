from keras.models import load_model
import numpy as np
from keras import backend as K
# from vnet_25D.vnet11_1.vnet import dice_coef


dir_num = 118
model_num = "6"
model = "tumour"

test_npy = "../dataset/vnet_3_1_test_npy/train/" + str(dir_num) + "/" + str(dir_num) + ".npy"
model_name = "./model_pre_mean/vnet_" + model + "_3_1_epoch" + model_num + ".hdf5"
predict_result = "./predict_npy_pre_mean/predict" + str(dir_num) + "_" + model \
                 + "_epoch" + model_num + ".npy"


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


# class WeightedBinaryCrossEntropy(object):
#
#     def __init__(self, pos_ratio=0.7):
#         neg_ratio = 1. - pos_ratio
#         self.pos_ratio = tf.constant(pos_ratio, tf.float32)
#         self.weights = tf.constant(neg_ratio / pos_ratio, tf.float32)
#         self.__name__ = "weighted_binary_crossentropy({0})".format(pos_ratio)
#
#     def __call__(self, y_true, y_pred):
#         return self.weighted_binary_crossentropy(y_true, y_pred)
#
#     def weighted_binary_crossentropy(self, y_true, y_pred):
#         # Transform to logits
#         epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
#         y_pred = tf.log(y_pred / (1 - y_pred))
#
#         cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.weights)
#         return K.mean(cost * self.pos_ratio, axis=-1)
# losses = WeightedBinaryCrossEntropy()


def test_model_fun():
    # 准备测试npy文件
    print("loading data")
    ############################################################
    imgs_test = np.load(test_npy)
    ############################################################
    imgs_test = imgs_test.astype('float32')
    imgs_test /= 255

    print("loading data done")
    # 加载模型
    ############################################################################################################
    # model = load_model(model_name, custom_objects={'dice_coef': dice_coef,
    #                                              'weighted_binary_crossentropy': losses.weighted_binary_crossentropy})
    model = load_model(model_name, custom_objects={'dice_coef': dice_coef})
    ############################################################################################################
    print("got unet")
    # 验证对于train的预测如何
    print('predict train_valid data')
    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
    ######################################################################################
    np.save(predict_result, imgs_mask_test)
    ######################################################################################


if __name__ == "__main__":
    test_model_fun()
