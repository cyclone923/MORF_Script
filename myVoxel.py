import math
import matplotlib.pyplot as plt
# the colormaps, grouped by category
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from sklearn.decomposition import PCA
import scipy.sparse

VOXESCALE = 2
THETA = 0.7
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


def elementVoxel(lab, *args):
    mass = readTypeInfo(lab)
    position = readPositionInfo(lab)

    orienting = PCA()
    orientedPosition = orienting.fit_transform(position)

    elementSequence = []
    classifiedNewPositions = {} #key: element#, value: a list of position
    classifiedOldPositions = {}
    for m, newp, oldp in zip(mass, orientedPosition, np.array(position)):
        elementNo = np.argmax(round(m) == elementMass)
        if elementNo not in classifiedNewPositions:
            classifiedNewPositions[elementNo] = []
            classifiedOldPositions[elementNo] = []
        classifiedNewPositions[elementNo].append(newp)
        classifiedOldPositions[elementNo].append(oldp)


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
    print("Meshgrid Size: "  +  str(size))

    print("Computing Grid")
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
                        else:
                            dataMatrix[elementNo][m][n][l] += calcVoxels(x - m, y - n, z - l, elementNo)
                        if dataMatrix[elementNo][m][n][l] > maxValue[elementNo]:
                            maxValue[elementNo] = dataMatrix[elementNo][m][n][l]
            maxIdx[0] = max(maxIdx[0], x + elementCutoff[elementNo])
            maxIdx[1] = max(maxIdx[1], y + elementCutoff[elementNo])
            maxIdx[2] = max(maxIdx[2], z + elementCutoff[elementNo])
            minIdx[0] = min(minIdx[0], x - elementCutoff[elementNo])
            minIdx[1] = min(minIdx[1], y - elementCutoff[elementNo])
            minIdx[2] = min(minIdx[2], z - elementCutoff[elementNo])
    dataMatrix = dataMatrix[:, minIdx[0]:maxIdx[0]+1, minIdx[1]:maxIdx[1]+1, minIdx[2]:maxIdx[2]+1]

    stat = dataMatrix.reshape(-1)
    # Just for Plotting.
    print("Ploting")
    fig = plt.figure(figsize=[27,18])
    fig.suptitle(lab+'-'+ args[0]+'\nCUTOFF='+str(CUTOFF)+'\nGaussian Parameter='+str(THETA)+'\nVoxel Size(Ang)='+str(VOXESCALE))

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.title.set_text('Original Position')
    for elementNo in sorted(classifiedOldPositions):
        oldPosition = np.stack(classifiedOldPositions[elementNo])
        ax.scatter(oldPosition[:, 0], oldPosition[:, 1], oldPosition[:, 2], color=color_scatter[elementNo])

    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.title.set_text('Oriented Position')
    for elementNo in sorted(classifiedNewPositions):
        newPosition = np.stack(classifiedNewPositions[elementNo])
        ax.scatter(newPosition[:, 0], newPosition[:, 1], newPosition[:, 2], color=color_scatter[elementNo])

    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.title.set_text('Voxels')


    colors = np.zeros(dataMatrix[0].shape + (4,))
    #should work for at most 4 element types
    for i in classifiedNewPositions:
        colors[..., 0] = color[i][0]
        colors[..., 1] = color[i][1]
        colors[..., 2] = color[i][2]
        colors[..., 3] = dataMatrix[i] / maxValue[i] + 0.1
        ax.voxels(dataMatrix[i], facecolors=colors)

    ax = fig.add_subplot(2, 3, 4)
    ax.title.set_text('Histogram')
    ax.hist(stat, bins=10)

    ax = fig.add_subplot(2, 3, 5)
    ax.title.set_text('Histogram (0.2~max)')
    ax.hist(stat, bins=100, range=(0.2, stat.max()))

    ax = fig.add_subplot(2, 3, 6)
    ax.title.set_text('xoy Section')
    subsection = np.zeros(size[0] * size[1] * elementTypes).reshape(elementTypes, size[0], size[1])
    for i in range(len(dataMatrix)):
        for m in range(len(dataMatrix[i])):
            for n in range(len(dataMatrix[i][0])):
                subsection[i][m][n] = dataMatrix[i][m][n][(-limit[2][0])]
    for i in range(len(subsection)):
        if args[0] != 'Wave Trans':
            ax.imshow(subsection[i], cmap=cmaps2[i], alpha=0.5)
        else:
            cmLimit = max([abs(min(stat)),max(stat)])
            ax.imshow(subsection[i], cmap=cmaps1[i],vmin=-cmLimit, vmax=cmLimit, alpha=0.5)

    plt.savefig('../voxel image/' + lab + '-' + args[0] + '-CUTOFF=' +str(CUTOFF) + '-' +str(THETA)+ '-' + str(VOXESCALE) + '.jpg')

    return np.sum(dataMatrix, axis=0), limit, elementSequence
    # return dataMatrix, limit, elementSequence




if __name__ == "__main__":

    rootPath = os.getcwd()
    for i in [0,201,498]:
        linkerDir = "linker" + str(i)
        os.chdir(rootPath + "/" + linkerDir + "_deformation")
        dataMatrix, _, _ = elementVoxel(linkerDir, 'Gaussian Only')
        print(dataMatrix.shape)
        np.save("voxel.npy", dataMatrix)


