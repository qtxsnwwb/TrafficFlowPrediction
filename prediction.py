import sys
import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import scipy.io as scio
import matplotlib.pyplot as plt

def transMatrixToTensor(data):
    """
    将矩阵转化为张量
    :param data: 待转化矩阵(12*365)
    :return: 转化后的张量(52*12*7)
    """
    time_steps = 7
    weekNum = int(data.shape[1]/time_steps)     #52
    inputTensor = np.zeros((weekNum, int(data.shape[0]), time_steps))     #52,12,7
    indexTemp = 0
    for i in range(0, weekNum*time_steps, time_steps):
        inputTensor[indexTemp, :, :] = data[:, i:i+time_steps]
        indexTemp += 1
    return inputTensor

def create_signMatrix(sign):
    """
    构建标签矩阵，将标签向量复制6次，组成12*7的矩阵
    :param sign: 标签向量
    :return: 标签矩阵(12*7)
    """
    signMatrix = np.zeros((len(sign), 7))   #12*7
    for i in range(7):
        signMatrix[:, i] = sign
    return signMatrix

def create_dataset(dataset, look_back):
    """
    切分数据集为训练数据和标签数据
    :param dataset: 原始数据集(12*365)
    :param look_back: 训练数据长度(相当于timestep)
    :return: 切分后的训练数据及标签数据(np.array)
    """
    timestep = look_back * 7
    dataX, dataY = [], []
    for i in range(dataset.shape[1]-timestep-1):
        tempMatrix = dataset[:, i:(i+timestep)]
        tempTensor = transMatrixToTensor(tempMatrix)
        dataX.append(tempTensor)
        tempMatrix = create_signMatrix(dataset[:, i+timestep])
        dataY.append(tempMatrix)
    return np.array(dataX), np.array(dataY)

def cnn_lstm_model(trainX, trainY, testX, testY):
    """
    CNN-LSTM预测模型
    :param trainX: 训练集训练数据
    :param trainY: 训练集标签
    :param testX: 测试集训练数据
    :param testY: 训练集标签
    :return: 训练后的模型
    """
    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), input_shape=(None, 12, 7, 1),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=False))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=1, kernel_size=(3, 3),
                     activation='sigmoid', padding='same',
                     data_format='channels_last'))
    model.compile(loss='mse', optimizer='adadelta')

    history = model.fit(trainX, trainY, batch_size=10, epochs=2000, validation_split=0.05)
    print(model.evaluate(testX, testY))

    # 保存模型
    model.save('model2.h5')

    # 画图
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model loss=MSE")
    plt.ylabel("Loss")
    plt.xlabel("Training Iteratins")
    plt.legend(["train", "val"], loc="lower right")
    plt.show()

def getDataset(data):
    """
    获取训练集和测试集
    :param data: 原始数据(12*365)
    :return: 训练集和测试集
    """
    # 原始数据转化为张量(52*12*7)
    dataTensor1 = transMatrixToTensor(data1)
    time_step = 2
    # 切分训练集与测试集
    train_size = int(data1.shape[1] * 0.7)
    trainlist = data1[:, :train_size]
    testlist = data1[:, train_size:]
    print(trainlist.shape, testlist.shape)
    trainX, trainY = create_dataset(trainlist, time_step)
    testX, testY = create_dataset(testlist, time_step)
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2], trainX.shape[3], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2], testX.shape[3], 1))
    trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], trainY.shape[2], 1))
    testY = np.reshape(testY, (testY.shape[0], testY.shape[1], testY.shape[2], 1))
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    return trainX, trainY, testX, testY



if __name__ == '__main__':
    # 获取2D-VMD分解后的矩阵（12*365）
    data = scio.loadmat("data1.mat")
    data1 = data["data1"]
    data = scio.loadmat("data2.mat")
    data2 = data["data2"]
    # 数据归一化
    max1 = np.max(data1)
    min1 = np.min(data1)
    max2 = np.max(data2)
    min2 = np.min(data2)
    data1 = (data1 - min1) / (max1 - min1)
    data2 = (data2 - min2) / (max2 - min2)
    # 切分数据为训练集和测试集
    trainX_1, trainY_1, testX_1, testY_1 = getDataset(data1)
    trainX_2, trainY_2, testX_2, testY_2 = getDataset(data2)
    # 运行CNN_LSTM模型
    #cnn_lstm_model(trainX_1, trainY_1, testX_1, testY_1)
    #cnn_lstm_model(trainX_2, trainY_2, testX_2, testY_2)
    #sys.exit(0)
    #导入模型
    model1 = load_model("model1.h5")
    testPrediction1 = model1.predict(testX_1)
    model2 = load_model("model2.h5")
    testPrediction2 = model2.predict(testX_2)
    #反归一化
    testY_1 = testY_1 * (max1 - min1) + min1
    testPrediction1 = testPrediction1 * (max1 - min1) + min1
    testY_2 = testY_2 * (max2 - min2) + min2
    testPrediction2 = testPrediction2 * (max2 - min2) + min2
    testY_result = testY_1 + testY_2
    testPrediction_result = testPrediction1 + testPrediction2
    #画图
    plt.figure(2)
    #plt.plot(testY_2[0, :, 0])
    #plt.plot(testPrediction2[0, :, 0])
    plt.subplot("421")
    plt.plot(testY_result[0, :, 0])
    plt.plot(testPrediction_result[0, :, 0])
    plt.subplot("422")
    plt.plot(testY_result[0, :, 1])
    plt.plot(testPrediction_result[0, :, 1])
    plt.subplot("423")
    plt.plot(testY_result[0, :, 2])
    plt.plot(testPrediction_result[0, :, 2])
    plt.subplot("424")
    plt.plot(testY_result[0, :, 3])
    plt.plot(testPrediction_result[0, :, 3])
    plt.subplot("425")
    plt.plot(testY_result[0, :, 4])
    plt.plot(testPrediction_result[0, :, 4])
    plt.subplot("426")
    plt.plot(testY_result[0, :, 5])
    plt.plot(testPrediction_result[0, :, 5])
    plt.subplot("427")
    plt.plot(testY_result[0, :, 6])
    plt.plot(testPrediction_result[0, :, 6])
    #plt.ylim((11,14))
    #plt.legend(["true", "prediction"])
    plt.show()



