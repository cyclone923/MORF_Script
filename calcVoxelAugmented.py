import math
import os
import numpy as np
from sklearn.decomposition import PCA
import sys

VOXESCALE = 2
THETA = 0.1
CUTOFF = 3
NUM_ELE = 4

elements = np.array(['H', 'C', 'N', 'O'])
elementMass = np.array([1, 12, 14, 16])
elementRadius = np.array([1.2, 1.7, 1.55, 1.52])  # Van der Waals radius
color = [[0, 0, 1], [1, 0.5, 0], [0, 1, 0], [1, 0, 0]]
color_scatter = ["blue", "orange", "green", "red"]
# H: blue, C: orange N: green O: red

elementTheta = THETA * elementRadius
elementOmega = 1.0 / elementTheta
elementCutoff = []
for e in elementTheta:
    elementCutoff.append(math.ceil(CUTOFF * e / VOXESCALE))
elementCutoff = np.array(elementCutoff)
systemCutoff = max(elementCutoff)


cmaps1 = ["PiYG", "PRGn", "PuOr", "RdGy"]
cmaps2 = ['Purples','Blues','Greens','Reds']


def readTypeInfo(lab):
    sour = open(lab + '.lmpdat', 'r')
    for line in sour:
        if line == 'Masses\n':
            mass_dict = {}
            while True:
                massLine = sour.readline()
                if massLine == '\n':
                    if mass_dict == {}:
                        continue
                    else:
                        break
                words = massLine.split()
                id = words[0]
                mass = words[1]
                mass_dict[id] = mass
        if line == 'Atoms\n':
            atomMasses = []
            while True:
                typeLine = sour.readline()
                if typeLine == '\n':
                    if atomMasses == []:
                        continue
                    else:
                        break
                words = typeLine.split()
                type = words[2]
                atomMasses.append(mass_dict[type])
    sour.close()
    atomMasses = list(map(float,atomMasses))
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
        position = list(map(float, words[1:]))
        positions.append(position)
    file.close()
    return positions


def calcVoxels(deltaX, deltaY, deltaZ, element):
    if deltaX == 0 and deltaY == 0 and deltaZ == 0 and elementTheta[element] == 0:
        return 1
    if elementTheta[element] == 0:
        return 0
    if (deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) > elementCutoff[element] ** 2:
        return 0

    return math.exp(-(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) * (VOXESCALE ** 2) / (2 * (elementTheta[element] ** 2)))


def waveTransform(deltaX, deltaY, deltaZ, element, *args):
    omega = elementOmega[element]
    return math.cos(2 * math.pi * omega * VOXESCALE * ((deltaX ** 2 + deltaY ** 2 + deltaZ ** 2) ** 0.5)) * calcVoxels(
        deltaX, deltaY, deltaZ, element, *args)

def rotate(theta, x, y):
    alpha = np.arctan(y/x)
    if x < 0:
        alpha += np.pi
    r = np.sqrt(x**2 + y**2)
    return r * np.cos(theta+alpha), r * np.sin(theta+alpha)

def elementVoxel(lab, *args):
    mass = readTypeInfo(lab)
    position = readPositionInfo(lab)

    orienting = PCA()
    orientedPosition = orienting.fit_transform(position)
    rotatedPosistion = [[], [], []]
    flippedPosition = [[], [], []]


    for i,(x,y,z) in enumerate(orientedPosition):
        newx, newy = rotate(2*np.pi*np.random.ranf(), x, y)
        rotatedPosistion[0].append([newx, newy, z])
        flippedPosition[0].append([-x, y, z])
        newy, newz = rotate(2*np.pi*np.random.ranf(), y, z)
        rotatedPosistion[1].append([x, newy, newz])
        flippedPosition[1].append([x, -y, z])
        newz, newx = rotate(2*np.pi*np.random.ranf(), z, x)
        rotatedPosistion[2].append([newx, y, newz])
        flippedPosition[2].append([x, y, -z])

    rotatedPosistion = list(map(np.array, rotatedPosistion))
    flippedPosition = list(map(np.array, flippedPosition))
    zeroflippedPosition = np.array(- orientedPosition)
    orientedPositions = np.stack([orientedPosition] + rotatedPosistion + flippedPosition + [zeroflippedPosition])

    all_matrix = []
    for orientedPosition in orientedPositions:
        elementSequence = []
        classifiedNewPositions = {} #key: element#, value: a list of position
        # classifiedOldPositions = {}
        for m, newp, oldp in zip(mass, orientedPosition, np.array(position)):
            elementNo = np.argmax(round(m) == elementMass)
            if elementNo not in classifiedNewPositions:
                classifiedNewPositions[elementNo] = []
                # classifiedOldPositions[elementNo] = []
            classifiedNewPositions[elementNo].append(newp)
            # classifiedOldPositions[elementNo].append(oldp)


        elementTypes = NUM_ELE#total element types=4
        size = [0, 0, 0]
        limit = np.array([[0, 0], [0, 0], [0, 0]])
        for i in range(len(size)):
            if max(orientedPosition[:, i]) > 0:
                limit[i][-1] = int(max(orientedPosition[:, i]) / VOXESCALE) + systemCutoff + 1
                size[i] += limit[i][-1]
            if min(orientedPosition[:, i]) < 0:
                limit[i][0] = int(min(orientedPosition[:, i]) / VOXESCALE) - systemCutoff - 1
                size[i] -= limit[i][0]
        dataMatrix = np.zeros(shape=(elementTypes, size[0], size[1], size[2]), dtype=np.float32)

        maxValue = [0 for _ in range(elementTypes)]
        minIdx = [size[i] for i in range(3)]
        maxIdx = [0 for _ in range(3)]
        for elementNo in sorted(classifiedNewPositions):#sequence by element#, H-C-N-0
            elementName = elements[elementNo]
            elementSequence.append(elementName)
            for point in classifiedNewPositions[elementNo]:
                x = int((point[0]) / VOXESCALE) - limit[0][0]
                y = int((point[1]) / VOXESCALE) - limit[1][0]
                z = int((point[2]) / VOXESCALE) - limit[2][0]
                for m in range(x - elementCutoff[elementNo], x + elementCutoff[elementNo] + 1):
                    for n in range(y - elementCutoff[elementNo], y + elementCutoff[elementNo] + 1):
                        for l in range(z - elementCutoff[elementNo], z + elementCutoff[elementNo] + 1):
                            if args[0] == 'Wave Trans':
                                dataMatrix[elementNo][m][n][l] += waveTransform(x - m, y - n, z - l, elementNo)
                            elif args[0] == 'Gaussian Only':
                                dataMatrix[elementNo][m][n][l] += calcVoxels(x - m, y - n, z - l, elementNo)
                            else:
                                print("Args[0] wrong, exit!")
                                exit(0)
                            if dataMatrix[elementNo][m][n][l] > maxValue[elementNo]:
                                maxValue[elementNo] = dataMatrix[elementNo][m][n][l]
                maxIdx[0] = max(maxIdx[0], x + elementCutoff[elementNo])
                maxIdx[1] = max(maxIdx[1], y + elementCutoff[elementNo])
                maxIdx[2] = max(maxIdx[2], z + elementCutoff[elementNo])
                minIdx[0] = min(minIdx[0], x - elementCutoff[elementNo])
                minIdx[1] = min(minIdx[1], y - elementCutoff[elementNo])
                minIdx[2] = min(minIdx[2], z - elementCutoff[elementNo])
        dataMatrix = dataMatrix[:, minIdx[0]:maxIdx[0]+1, minIdx[1]:maxIdx[1]+1, minIdx[2]:maxIdx[2]+1]
        all_matrix.append(dataMatrix)


    return all_matrix

'''
for each molecule, return 8 3-D matrix representing the flowing:
1. Orginal matrix
2. Rotate in xoy plane with random (0,2*pi)
3. Rotate in yoz plane with random (0,2*pi)
4. Rotate in zox plane with random (0,2*pi)
5. Flip x cordinate with repect to origin
6. Flip y cordinate with repect to origin
7. Flip z cordinate with repect to origin
8. Flip x,y,z cordinate with repect to origin
'''


if __name__ == "__main__":
    maxX = 0
    maxY = 0
    maxZ = 0

    # # d = sys.argv[1]
    start = sys.argv[1]
    end = sys.argv[2]
    rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/BFS/finalNode/depth5" #d4 302-6394 d5 6395-110000 d5Unused 110000-170000
    # for linkerDir in os.listdir(rootPath):
    for i in range(int(start),int(end)):
        linkerDir = "linker" + str(i)
        os.chdir(rootPath + "/" + linkerDir + "/" + linkerDir + "_deformation")
        lab = linkerDir
        all_matrix = elementVoxel(lab, 'Gaussian Only')
        if "augmentVoxel" in os.listdir(os.getcwd()):
            os.rename("augmentVoxel", "augmentVoxel7")
        if "augmentVoxel1" not in os.listdir(os.getcwd()):
            os.mkdir("augmentVoxel1")
        for i,x in enumerate(all_matrix):
            np.save("augmentVoxel1/" + str(i) + ".npy", x)
            maxX = max(maxX, x.shape[1])
            maxY = max(maxY, x.shape[2])
            maxZ = max(maxZ, x.shape[3])


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
    #                 dataMatrix, _, _ = elementVoxel(lab, 'Wave Trans')
    #                 np.save("voxel.npy", dataMatrix)
    #                 maxX = max(maxX, dataMatrix.shape[1])
    #                 maxY = max(maxY, dataMatrix.shape[2])
    #                 maxZ = max(maxZ, dataMatrix.shape[3])

    print(maxX, maxY, maxZ)







