import math
import os
import numpy as np
from sklearn.decomposition import PCA
import sys


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

def calcPointCloud(lab):
    mass_ele = {1:1, 12:2, 14:3, 16:4}
    all_mass = list(map(lambda x: round(x), readTypeInfo(lab)))
    position = readPositionInfo(lab)

    position = np.array(position)
    mass = np.zeros(shape=(len(all_mass), 5))
    for i,x in enumerate(all_mass):
        mass[i][mass_ele[x]] = 1
    positionWithMass = np.concatenate((position, mass), axis=1)
    positionWithMass = np.concatenate((positionWithMass, np.array([[0,0,0,1,0,0,0,0]])), axis=0)
    return positionWithMass

if __name__ == "__main__":
    maxX = 0


    # # d = sys.argv[1]
    start = sys.argv[1]
    end = sys.argv[2]
    rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/BFS/finalNode/depth5" #d4 302-6394 d5 6395-110000 d5Unused 110000-170000
    # for linkerDir in os.listdir(rootPath):
    for i in range(int(start),int(end)):
        linkerDir = "linker" + str(i)
        os.chdir(rootPath + "/" + linkerDir + "/" + linkerDir + "_deformation")
        lab = linkerDir
        x = calcPointCloud(lab)
        np.save("pointCloud.npy", x)
        maxX = max(maxX, x.shape[0])



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

    print(maxX)