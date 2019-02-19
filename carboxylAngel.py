
import numpy as np
import math
import os
import sys

def readOneFrame(file):
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
    return np.array([positions[0],positions[2],positions[-2],positions[-1]])

def calcNewCoor(positions):
    nodes1 = [positions[0],positions[1]]
    nodes2 = positions[-2:]
    centers = np.array([sum(np.array(nodes1))/2,sum(np.array(nodes2))/2])
    x = centers[0]-centers[1]
    x = x / np.dot(x,x)**0.5
    nodes2 = np.array(nodes2)
    nodes1 = np.array(nodes1)
    directions = np.array([nodes1[0]-nodes1[1],nodes2[0]-nodes2[1]])
    if np.dot(directions[0],directions[1]) >= 0:
        y = directions[0] + directions[1]
    else:
        y = directions[0] - directions[1]
    z = np.cross(x,y)
    z = z / np.dot(z,z)**0.5
    y = np.cross(z,x)
    transMat = np.mat(np.matrix([x,y,z]))
    return transMat

def newPosition(position, coor):
    transPosition = np.array((coor * np.mat(position).T).T).flatten()
    return transPosition

def calcDirections(positions):
    transMatrix = calcNewCoor(positions)
    for i in range(len(positions)):
        positions[i] = newPosition(positions[i], transMatrix)
    #print (positions)
    return [positions[0]-positions[1],positions[-1]-positions[-2]]

def angle(lab):
    positionsFile = open(lab + '-un-squashed.xyz', 'r')
    positionsFile.readline()
    positionsFile.readline()
    positions = readOneFrame(positionsFile)
    #positions = [[0,1,1],[1,1.414,2],[5,1,2],[6,2,2.5]]
    res = calcDirections(positions)
    if abs(np.dot(res[0][:2], res[1][:2])/np.dot(res[0][:2], res[0][:2])**0.5/np.dot(res[1][:2], res[1][:2])**0.5) > 1:
        return 180
    angel = math.acos(np.dot(res[0][:2], res[1][:2])/np.dot(res[0][:2], res[0][:2])**0.5/np.dot(res[1][:2], res[1][:2])**0.5)
    if angel > math.pi/2:
        return angel/math.pi*180
    else:
        return 180 - angel / math.pi * 180


if __name__ == "__main__":



    # # d = sys.argv[1]
    start = sys.argv[1]
    end = sys.argv[2]
    rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/BFS/finalNode/depth5" #d4 302-6394 d5 6395-110000 d5Unused 110000-170000
    # for linkerDir in os.listdir(rootPath):
    for i in range(int(start),int(end)):
        linkerDir = "linker" + str(i)
        os.chdir(rootPath + "/" + linkerDir + "/" + linkerDir + "_deformation")
        lab = linkerDir
        x = angle(lab)
        np.save("carboxAngle.npy", x)
