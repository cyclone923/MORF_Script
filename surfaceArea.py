# Python 3
# Pull out information from linker{#}-ave-force.d, Calculate the needed results, fit the data to a function and shift
# the curve through zero point



import math
import numpy.polynomial.polynomial as poly
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
# the colormaps, grouped by category
import scipy.interpolate as inter
from mpl_toolkits.mplot3d.axes3d import get_test_data
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import numpy as np
from sklearn.decomposition import PCA

voxelScale = 0.1
theta = 0.5
cutoff = 3

elements = np.array(['C', 'H', 'O', 'N', 'Si'])
elementMass = np.array([12, 1, 16, 14, 28])
elementRadius = np.array([1.7, 1.2, 1.52, 1.55, 2.1])  # Van der Waals radius

elementTheta = theta * elementRadius
elementOmega = 1.0 / elementTheta
elementCutoff = []
for e in elementTheta:
    elementCutoff.append(math.ceil(cutoff * e / voxelScale))
elementCutoff = np.array(elementCutoff)
systemCutoff = max(elementCutoff)

color = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
cmaps1 = ["PiYG", "PRGn", "PuOr", "RdGy"]
cmaps2 = ['Purples','Blues','Greens','Reds']

def readTypeInfo(lab):
    sour = open(lab + '.lmpdat', 'r')
    for line in sour:
        if line == '\n':
            continue
        else:
            if line == 'Masses\n':
                mass = []
                while True:
                    try:
                        massLine = sour.readline()
                        if massLine == '\n':
                            continue
                        words = massLine.split()
                        int(words[0])
                        mass.append(words[0:2])
                        mass[-1][0] = str(mass[-1][0])
                    except:
                        break
                line = massLine
            if line == 'Atoms\n':
                type = []
                while True:
                    try:
                        typeLine = sour.readline()
                        if typeLine == '\n':
                            continue
                        words = typeLine.split()
                        int(words[0])
                        type.append([words[0], words[2]])
                    except:
                        break
                line = typeLine
    sour.close()
    atomMasses = []
    for e in type:
        for element in mass:
            if e[-1] == element[0]:
                atomMass = float(element[-1])
        atomMasses.append(atomMass)
    return atomMasses

def readPositionInfo(lab):
    file = open(lab + '-un-squashed.xyz', 'r')
    file.readline()
    file.readline()
    positions = []
    for line in file:
        if line == '\n':
            break
        words = line.split()
        position = words[1:]
        while type(position[0]) is str:
            position.append(float(position[0]))
            position.pop(0)
        positions.append(position)
    file.close()
    return positions

def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

def calcVoxels(deltaX, deltaY, deltaZ, element, *args):
    if deltaX == 0 and deltaY == 0 and deltaZ == 0 and elementTheta[element] == 0:
        return 1
    if elementTheta[element] == 0:
        return 0
    if len(args)>0:
        return math.exp(
            -(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) * (voxelScale ** 2) / (2 * (elementTheta[element] ** 2)))
    if (deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) > elementCutoff[element] ** 2:
        return 0
    return math.exp(-(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) * (voxelScale ** 2) / (2 * (elementTheta[element] ** 2)))

# def waveTransform(deltaX, deltaY, deltaZ, element, *args):
#     omega = elementOmega[element]
#     return math.cos(2 * math.pi * omega * voxelScale * ((deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) ** 0.5)) * calcVoxels(
#         deltaX, deltaY, deltaZ, element, *args)

def surfaceArea(mass, position):
    orienting = PCA()
    orientedPosition = orienting.fit_transform(position)
    position.clear()
    classifiedMasses = []
    elementSequence = []
    classifiedPositions = []
    orientedPosition = np.array(orientedPosition)
    for i in range(len(mass)):
        if mass[i] not in classifiedMasses:
            classifiedMasses.append(mass[i])
            classifiedPositions.append([orientedPosition[i]])
        else:
            classifiedPositions[classifiedMasses.index(mass[i])].append(orientedPosition[i])
    classifiedPositions = np.array(classifiedPositions)

    size = [0, 0, 0]
    limit = np.array([[0, 0], [0, 0], [0, 0]])
    for i in range(len(size)):
        if max(orientedPosition[:, i]) > 0:
            limit[i][-1] = int(max(orientedPosition[:, i]) / voxelScale) + systemCutoff + 1
            size[i] += limit[i][-1]
        if min(orientedPosition[:, i]) < 0:
            limit[i][0] = -(int(abs(min(orientedPosition[:, i])) / voxelScale) + systemCutoff + 1)
            size[i] -= limit[i][0]
    dataMatrix = np.zeros(size[0] * size[1] * size[2]).reshape(size[0], size[1], size[2])
    maxValue = 0
    for i in range(len(classifiedMasses)):
        elementNo = np.where(elementMass == round(classifiedMasses[i]))[0][0]
        elementSequence.append(elements[elementNo])
        for e in classifiedPositions[i]:
            x = int((e[0]) / voxelScale) - limit[0][0]
            y = int((e[1]) / voxelScale) - limit[1][0]
            z = int((e[2]) / voxelScale) - limit[2][0]
            for m in range(x - elementCutoff[elementNo], x + elementCutoff[elementNo] + 1):
                for n in range(y - elementCutoff[elementNo], y + elementCutoff[elementNo] + 1):
                    for l in range(z - elementCutoff[elementNo], z + elementCutoff[elementNo] + 1):
                        dataMatrix[m][n][l] += calcVoxels(x - m, y - n, z - l, elementNo)


    stat = [0,0,0,0,0,0,0]
    for m in range(len(dataMatrix)):
        for n in range(len(dataMatrix[0])):
            for l in range(len(dataMatrix[0][0])):
                if dataMatrix[m][n][l] >= math.e**-2:
                    i = 0
                    if dataMatrix[m-1][n][l] < math.e**-2:
                        i+=1
                    if dataMatrix[m+1][n][l] < math.e**-2:
                        i+=1
                    if dataMatrix[m][n-1][l] < math.e**-2:
                        i+=1
                    if dataMatrix[m][n+1][l] < math.e**-2:
                        i+=1
                    if dataMatrix[m][n][l-1] < math.e**-2:
                        i+=1
                    if dataMatrix[m][n][l+1] < math.e**-2:
                        i+=1
                    stat[i]+=1
    stat = np.array(stat)
    facesArea = [0,1,2**0.5,3**0.5,2**0.5,1,0]
    surface = sum([stat[i]*facesArea[i] for i in range(len(stat))])*voxelScale**2

    return surface

def readMoleculeSurfaceArea(lab, *args):
    mass = readTypeInfo(lab)
    position = readPositionInfo(lab)
    return surfaceArea(mass, position)



if __name__ == "__main__":
    # d = sys.argv[1]
    start = sys.argv[1]
    end = sys.argv[2]
    rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/BFS/finalNode/depth4"  #d4 302-6394 d5 6395-171390
    # for linkerDir in os.listdir(rootPath):
    for i in range(int(start),int(end)):
        linkerDir = "linker" + str(i)
        os.chdir(rootPath + "/" + linkerDir + "/" + linkerDir + "_deformation")
        lab = linkerDir
        s = readMoleculeSurfaceArea(lab)
        np.save(lab + '-surfaceArea.npy', s)



    # rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/candidate"
    # start = int(sys.argv[1])
    # end = int(sys.argv[2])

    # for t in range(start, end):
    #     for d in range(1, 31):
    #         depthDir = rootPath + "/" + str(t) + "/Depth" + str(d)
    #         for dir in os.listdir(depthDir):
    #             if dir[-11:] == "deformation":
    #                 os.chdir(depthDir + "/" + dir)
    #                 lab = dir[:-12]
    #                 s = readMoleculeSurfaceArea(lab)
    #                 np.save(lab + '-surfaceArea.npy', s)



