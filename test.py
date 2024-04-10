import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from typing import *
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def fileMatrix(filename):
    """ 数据读取与转换 """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numberOfLines,3))
    #返回的分类标签向量
    classLabelVector = []
    #行的索引值
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        #根据文本中标记的喜欢程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


def showdatas(datingDataMat, datingLabels):
    """ 数据的二维图表显示 """
    # 设置汉字格式
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False,
                            sharey=False, figsize=(13,8))
    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('red')
        if i == 2:
            LabelsColors.append('green')
        if i == 3:
            LabelsColors.append('blue')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、
    # 第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'Flyer Miles Per Year Compared To Playing Video Games')
    axs0_xlabel_text = axs[0][0].set_xlabel(u'Flyer Miles Earned Each Year')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'Time Spent Playing Video Games')
    plt.setp(axs0_title_text, size=10, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=10, weight='bold', color='black')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、
    # 第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'Flyer Miles Per Year And Ice Cream Per Week')
    axs1_xlabel_text = axs[0][1].set_xlabel(u'Flyer Miles Earned Each Year')
    axs1_ylabel_text = axs[0][1].set_ylabel(u'Ice Cream Consumed Per Week')
    plt.setp(axs1_title_text, size=10, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=10, weight='bold', color='black')
    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、
    # 第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title( u'Percentage Of Playing Video Games And Ice Cream Consumed Per Week')
    axs2_xlabel_text = axs[1][0].set_xlabel( u'Percentage Of Time Spent Playing Video Games')
    axs2_ylabel_text = axs[1][0].set_ylabel( u'Litres Of Ice Cream Consumed Per Week')
    plt.setp(axs2_title_text, size=10, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=10, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='green', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='blue', marker='.', markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    # 调整子图间距
    plt.subplots_adjust(wspace =0.2, hspace =0.4)
    # 显示图片
    plt.show()



def show_dim3(data, labels):
    """ 三维图像 """
    # 特征归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # 建立三维坐标系并绘制空间散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = {1: 'red', 2: 'green', 3: 'blue'}
    likes = {
        1: "didntLike",
        2: "smallDoses",
        3: "largeDoses"
    }

    for label in np.unique(labels):
        indices = np.where(labels == label)
        ax.scatter(scaled_data[indices, 0], scaled_data[indices, 1], scaled_data[indices, 2], color=colors[label], 
        
        label=str(likes[label]))

    ax.set_xlabel('Flyer Miles')
    ax.set_ylabel('Play games')
    ax.set_zlabel('Ice-cream')

    # 显示图例
    ax.legend()
    plt.show()


# 打开的文件名
filename = "datingTestSet.txt"
# 打开并处理数据
datas, labels = fileMatrix(filename)
# 二维图表图像
showdatas(datas, labels)
# 三维图像
show_dim3(datas, labels)


copy_datas = datas
copy_labels = labels

# 训练数据
copy_datas = np.array(copy_datas)
copy_labels = np.array(copy_labels)

k_nn_min = 1
k_nn_max = 30
k_fold_min = 5
k_fold_max = 20

k_ns = [i for i in range(k_nn_min, k_nn_max, 2)]
k_folds = [k for k in range(k_fold_min, k_fold_max)]
accus = []

for k in k_ns:
    accuracies = []
    for k_fold in k_folds:
        knn = KNeighborsClassifier(n_neighbors=k)
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        scores = cross_val_score(knn, copy_datas, copy_labels, cv=kf, scoring='accuracy')
        accuracies.append([np.mean(scores), np.std(scores)])
    accus.append(accuracies)


# 创建 3D 图表
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建网格
X, Y = np.meshgrid(k_folds, k_ns)

# 提取准确度均值数据
Z = np.array([[item[0] for item in row] for row in accus])

# 绘制三维曲面图
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# 添加轴标签和标题
ax.set_xlabel('K Folds')
ax.set_ylabel('K Nearest Neighbors')
ax.set_zlabel('Accuracy Mean')
ax.set_title('Accuracy Mean in 3D Space')

# 显示图表
plt.show()