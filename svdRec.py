# coding=gbk
'''
Created on 2015年4月7日

@author: Blunce
'''

import numpy as np


def loadData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    
def eulidSim(inA, inB):
    return 1.0 / (1.0 + np.linalg.norm(inA, inB)) 

def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    else:
        return 0.5 + 0.5 * (np.corrcoef(inA, inB, rowvar=0)[0][1])
    
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)

def standEst(datMat, user, simMeans, item):
    n = np.shape(datMat)[1]
    simTotal = 0.0
    ratSimTatal = 0.0
    for j in range(n):
        userRating = datMat[user, j]
        if userRating == 0:
            continue
        overLap = np.nonzero(np.logical_and(datMat[:, item].A > 0, datMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeans(datMat[overLap, item], datMat[overLap, j])
        simTotal += similarity
        ratSimTatal += similarity * userRating
    if simTotal == 0.0:
        return 0
    else:
        return ratSimTatal / simTotal
    
def recommend(dataMat, user, N=3, simMeans=cosSim, estMeathod=standEst):
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rate everything'
    itemSores = []
    for item in unratedItems:
        estimatedSore = estMeathod(dataMat, user, simMeans, item)
        itemSores.append((item, estimatedSore))
    return sorted(itemSores, key=lambda jj:jj[1], reverse=True)[:N]

def svdEst(dataMat, user, simMeans, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = np.linalg.svd(dataMat)
    Sig4 = np.mat(np.eye(4) * Sigma[:4])
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeans(xformedItems[item, :].T, xformedItems[j, :].T)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal
    
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print 1,
            else:
                print 0,
        print ''
        
def imgCompress(numSv=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print "****original matrix******"
    printMat(myMat, thresh)
    U, sigma, VT = np.linalg.svd(myMat)
    SigRecon = np.mat(np.zeros((numSv, numSv)))
    for k in range(numSv):
        SigRecon[k, k] = sigma[k]
    reconMat = U[:, :numSv] * SigRecon * VT[:numSv, :]
    print "****reconstructed matrix*************"
    printMat(reconMat, thresh)
    return myMat, reconMat
     
def countOne(matrix):
    count = 0
    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[1]):
            if matrix[i, j] == 1:
                count += 1
    return count

def NormalSVD(matrix, thread=0.9):
    U, Sigma, VT = np.linalg.svd(matrix)
    Sig2 = Sigma ** 2
    sumPower = sum(Sig2)
    num = 0
    for num in range(len(Sig2)):
        if sum(Sig2[:num]) >= thread:
            break
    SigRecon = np.mat(np.zeros((num, num)))
    for k in range(num):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:,:num]*SigRecon*VT[:num,:]
    return reconMat
