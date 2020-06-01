import matplotlib.pyplot as plt
import numpy as np

title = "dice_list"
dice_list = [1, 2, 3, 4, 5, 11, 11, 22, 33, 33, 32, 3, 55, 44, 4, 57, 6, 7, 57, 66, 66, 66, 66, 88, 88, 88, 88, 99, 99, 91]


def plot_test_dice_histogram(dice_list):
    dice_array = np.array(dice_list)
    dice_90 = len(dice_array[90 <= dice_array])
    dice_80 = len(dice_array[80 <= dice_array]) - len(dice_array[90 <= dice_array])
    dice_70 = len(dice_array[70 <= dice_array]) - len(dice_array[80 <= dice_array])
    dice_60 = len(dice_array[60 <= dice_array]) - len(dice_array[70 <= dice_array])
    dice_50 = len(dice_array[50 <= dice_array]) - len(dice_array[60 <= dice_array])
    dice_40 = len(dice_array[40 <= dice_array]) - len(dice_array[50 <= dice_array])
    dice_30 = len(dice_array[30 <= dice_array]) - len(dice_array[40 <= dice_array])
    dice_20 = len(dice_array[20 <= dice_array]) - len(dice_array[30 <= dice_array])
    dice_10 = len(dice_array[10 <= dice_array]) - len(dice_array[20 <= dice_array])
    dice_0 = len(dice_array[0 <= dice_array]) - len(dice_array[10 <= dice_array])
    Y = np.array([dice_0, dice_10, dice_20, dice_30, dice_40, dice_50, dice_60, dice_70, dice_80, dice_90])
    X = np.arange(10)
    print(X, Y)

    plt.bar(10*X, +Y, facecolor='#9999ff', edgecolor='white')
    # 这里zip的作用是每一步输出两个值，X和Y1给x，y
    for x, y in zip(X, Y):
        # ha:horizontal alignment
        plt.text(10*x, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_test_dice_histogram(dice_list)
