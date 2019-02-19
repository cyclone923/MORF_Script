
import sys
import os
import math
import numpy as np
from copy import copy

def calcLinkMat(bondmat):
    atomNo = len(bondmat)
    result = bondmat.copy()
    for i in range(atomNo):
        for j in range(atomNo):
            for k in range(atomNo):
                if result[j][k] > result[j][i] + result[i][k]:
                    result[j][k] = result[j][i] + result[i][k]
    return result

def distanceMat(lab):

    iniDis = float('inf')

    f = open(lab + '.lmpdat', 'r')
    # f = open(lab+'.coeff', 'r')
    for line in f:
        if line == 'Bonds\n':
            f.readline()
            break
    links = []
    for line in f:
        if line == '\n':
            break
        info = line.split()
        links.append([int(info[2]), int(info[3])])
    f.close()
    atomNo = max(sum(links, []))
    linkMat = [0] * atomNo
    for i in range(atomNo):
        linkMat[i] = [float('inf')] * atomNo
        linkMat[i][i] = 0
    for e in links:
        linkMat[e[0] - 1][e[1] - 1] = 1
        linkMat[e[1] - 1][e[0] - 1] = 1
    linkMat = np.array(linkMat)
    distanceMat = calcLinkMat(linkMat)
    shortestDis = distanceMat[0][-1]
    shortestPath = []
    backbone = []
    for i in range(len(distanceMat)):
        if distanceMat[i][0]+distanceMat[i][-1]==shortestDis:
            shortestPath.append(i)
            backbone.append(i)


    for e1 in shortestPath:
        for e2 in shortestPath:
            if linkMat[e1][e2] != 1 or e2<e1:
                continue
            delLinkMat = linkMat.copy()
            delLinkMat[e1][e2] = float('inf')
            delLinkMat[e2][e1] = float('inf')
            distanceMat = calcLinkMat(delLinkMat)
            shortestDis = distanceMat[0][-1]
            if shortestDis != float('inf'):
                for i in range(len(distanceMat)):
                    if i not in backbone:
                        if distanceMat[i][0] + distanceMat[i][-1] == shortestDis:
                            backbone.append(i)
    armSites = []

    for e in backbone:
        if e not in shortestPath:
            shortestPath.append(e)
    for i in range(atomNo):
        if i in shortestPath:
            continue
        for e in shortestPath:
            if linkMat[i][e] != 1:
                continue
            counts = 0
            for element in linkMat[i]:
                if element == 1:
                    counts+=1
            if counts == 1 and i not in backbone:
                backbone.append(i)
            if counts > 1 and i not in armSites:
                armSites.append(i)

    backbone.sort()
    for i in range(len(backbone)):
        backbone[i] += 1
    for i in range(len(armSites)):
        armSites[i] += 1
    #print(backbone,len(backbone),armSites,len(backbone))

    return (atomNo-len(backbone))/len(backbone)


# # d = sys.argv[1]
start = sys.argv[1]
end = sys.argv[2]
rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/BFS/finalNode/depth5" #d4 302-6394 d5 6395-110000 d5Unused 110000-170000
# for linkerDir in os.listdir(rootPath):
for i in range(int(start),int(end)):
    linkerDir = "linker" + str(i)
    os.chdir(rootPath + "/" + linkerDir + "/" + linkerDir + "_deformation")
    lab = linkerDir
    x = distanceMat(lab)
    print(linkerDir)
    print(x)
    np.save("bbRatio.npy", x)
