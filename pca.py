# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:48:23 2018

@author: 游侠-Speed
"""

import numpy as np

def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    return newData,meanVal

percentage = float(input("想要新的特征能表达原来特征的百分之多少（用小数表示）： "))

def pca(dataMat,percentage=percentage):  
    newData,meanVal=zeroMean(dataMat)  
    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本  
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))    #求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量  
    n=percentage2n(eigVals,percentage)                 #要达到percent的方差百分比，需要前n个特征向量  
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序  
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标  
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量  
    lowDDataMat=newData*n_eigVect               #低维特征空间的数据  
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据  
    return lowDDataMat,reconMat  


def percentage2n(eigVals,percentage):  
    sortArray=np.sort(eigVals)   #升序  
    sortArray=sortArray[-1::-1]  #逆转，即降序  
    arraySum=sum(sortArray)  
    tmpSum=0  
    num=0  
    for i in sortArray:  
        tmpSum+=i  
        num+=1  
        if tmpSum>=arraySum*percentage:  
            return num  

import xlrd


def read(file, sheet_index=0):
    """

    :param file: 文件路径
    :param sheet_index: 读取的工作表索引
    :return: 二维数组
    """
    workbook = xlrd.open_workbook(file)
    # all_sheets_list = workbook.sheet_names()
    # print("本文件中所有的工作表名称:", all_sheets_list)
    # 按索引读取工作表
    sheet = workbook.sheet_by_index(sheet_index)
    print(sheet)
    print("工作表名称:", sheet.name)
    print("行数:", sheet.nrows)
    print("列数:", sheet.ncols)

    # 按工作表名称读取数据
    # second_sheet = workbook.sheet_by_name("b")
    # print("Second sheet Rows:", second_sheet.nrows)
    # print("Second sheet Cols:", second_sheet.ncols)
    # 获取单元格的数据
    # cell_value = sheet.cell(1, 0).value
    # print("获取第2行第1列的单元格数据:", cell_value)
    
    data = []
    for i in range(0, sheet.nrows):
        data.append(sheet.row_values(i))
    return data


if __name__ == '__main__':

    dataMat = read('数据.xlsx')
    lowDDataMat,reconMat = pca(dataMat)
    print(lowDDataMat)







