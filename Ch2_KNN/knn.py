from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 分类器
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet     # tile将inX 1*2 变为 4*2的矩阵(复制)
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)               # 对平方差的矩阵的每一行计算求和
    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()            # 返回的是从小到大的索引值
    classCount = {}                                     # 字典类型，每个标签出现的个数

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]      # 距离最小的所属类型
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)     # 按照第二个元素进行排序,逆序

    return sortedClassCount[0][0]

# 文件转numpy矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0

    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    
    return returnMat, classLabelVector

# 归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)        # 选取列的最小值，是一个1*3
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))   # 重复为m*3的矩阵
    normDataSet = normDataSet / tile(ranges, (m,1))

    return normDataSet, ranges, minVals

# 约会数据测试
def datingClassTest():
    hoRatio = 0.10       # 划分训练集和测试集的比例
    datingDataMat, datingLabels = file2matrix('Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)      # 测试集数目
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs : m, :], datingLabels[numTestVecs : m], 3) 
        print("the classifier came back with : %d, the real answer is : %d " % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0

    print("the total error rate is: %f " % (errorCount / float(numTestVecs)))


# 手写数字识别，图像转向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)

    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


if __name__ == "__main__":
    datingClassTest()