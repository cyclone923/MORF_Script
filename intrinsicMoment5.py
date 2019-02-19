
import sys
import os
import numpy as np
import copy
import time

orders = ['0th','1th','2nd','3rd','4th','5th']
def readMom(file, order, *args):
    for line in file:
        if line == '\n':
            continue
        words = line.split(':')
        if words[0] == orders[order]:
            if order == 0:
                mom = float(words[-1].split()[-1])
            if order == 1:
                mom = float(words[-1].split()[args[0]])
            if order == 2:
                for i in range(args[0]+1):
                    mom = float(file.readline().split()[args[1]])
            if order == 3:
                for i in range(args[0]):
                    for count in range(4):
                        file.readline()
                for i in range(args[1]+1):
                    mom = float(file.readline().split()[args[2]])
            if order == 4:
                for i in range(args[0]):
                    for count in range(13):
                        file.readline()
                for i in range(args[1]):
                    for count in range(4):
                        file.readline()
                for i in range(args[2]+1):
                    mom = float(file.readline().split()[args[3]])
            if order == 5:
                for i in range(args[0]):
                    for count in range(40):
                        file.readline()
                for i in range(args[1]):
                    for count in range(13):
                        file.readline()
                for i in range(args[2]):
                    for count in range(4):
                        file.readline()
                for i in range(args[3]+1):
                    mom = float(file.readline().split()[args[4]])

    return mom

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
    return positions

def calcMassCenter(masses, positions):
    massCenter = [sum(x) for x in np.matrix.transpose(np.array([i * j for i, j in zip(np.array(masses), np.array(positions))]))] / sum(np.array(masses))
    return massCenter

def calcNewCoor(positions):
    nodes1 = [positions[0],positions[2]]
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

def newPosition(position, coor, origin):
    transPosition = np.array((coor * np.mat(position - origin).T).T).flatten()
    return transPosition

def calcMoment(masses, positions):
    # all the moments are normalized
    zerothMoment = sum(np.array(masses))
    firstMoment = calcMassCenter(masses, positions)
    transMatrix = calcNewCoor(positions)
    for i in range(len(positions)):
        positions[i] = newPosition(positions[i], transMatrix,firstMoment)

    secondMoment = np.array(sum([(x / zerothMoment *
                                  np.outer(np.matrix.transpose(y),(y)))for x, y in zip(masses, positions)]))

    eigenSecondMoment = np.linalg.eigvals(secondMoment)

    thirdMoment = []
    temp = thirdMoment.copy()
    for order in range(3):
        for i in range(3):
            thirdMoment.append(temp)
        temp = thirdMoment.copy()
        thirdMoment.clear()
    thirdMoment = temp.copy()
    temp.clear()
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                thirdMoment[i1][i2][i3] = 0.0
    thirdMoment = np.array(thirdMoment)
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                thirdMoment[i1][i2][i3]=(sum([(mass / zerothMoment * position[i1] * position[i2] * position[i3]) for mass, position in zip(masses, positions)]))

    fourthMoment = []
    temp = fourthMoment.copy()
    for order in range(4):
        for i in range(3):
            fourthMoment.append(temp)
        temp = fourthMoment.copy()
        fourthMoment.clear()
    fourthMoment = temp.copy()
    temp.clear()
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    fourthMoment[i1][i2][i3][i4] = 0.0
    fourthMoment = np.array(fourthMoment)
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    fourthMoment[i1][i2][i3][i4] = (sum([(mass / zerothMoment * position[i1] * position[i2] * position[i3] * position[i4]) for mass, position in zip(masses, positions)]))

    fifthMoment = []
    temp = fifthMoment.copy()
    for order in range(5):
        for i in range(3):
            fifthMoment.append(temp)
        temp = fifthMoment.copy()
        fifthMoment.clear()
    fifthMoment = temp.copy()
    temp.clear()
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        fifthMoment[i1][i2][i3][i4][i5] = 0.0
    fifthMoment = np.array(fifthMoment)
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        fifthMoment[i1][i2][i3][i4][i5] = (sum(
                            [(mass / zerothMoment * position[i1] * position[i2] * position[i3] * position[i4] * position[i5]) for
                             mass, position in zip(masses, positions)]))
    return zerothMoment, firstMoment, secondMoment, eigenSecondMoment, thirdMoment, fourthMoment, fifthMoment

def pullMomentforSteps(lab):
    moments = []
    for i in range(6):
        moments.append([])
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        temp = [i1, i2, i3, i4, i5]
                        temp.sort()
                        if temp not in moments[5]:
                            moments[5].append(temp)
                    temp = [i1, i2, i3, i4]
                    temp.sort()
                    if temp not in moments[4]:
                        moments[4].append(temp)
                temp = [i1, i2, i3]
                temp.sort()
                if temp not in moments[3]:
                    moments[3].append(temp)
            temp = [i1, i2]
            temp.sort()
            if temp not in moments[2]:
                moments[2].append(temp)
        temp = [i1]
        temp.sort()
        if temp not in moments[1]:
            moments[1].append(temp)
    moments[0].append([])
    data = []
    for i in range(6):
        for e in moments[i]:
            momentsFile = open(lab + '-intrinsic-moments.txt', 'r')
            momentsFile.readline()
            data.append(readMom(momentsFile, i, *e))
    return data

def intriMoment(lab):
    positionsFile = open(lab + '-un-squashed.xyz', 'r')
    positionsFile.readline()
    positionsFile.readline()
    mass = readTypeInfo(lab)
    mt = []
    while True:
        try:
            position = readOneFrame(positionsFile)
            zerothMoment, firstMoment, secondMoment, eigenSecondMoment, thirdMoment, fourthMoment, fifthMoment = calcMoment(mass, position)
            mt.append([0,zerothMoment, firstMoment, secondMoment, eigenSecondMoment, thirdMoment, fourthMoment, fifthMoment])

        except:
            break
    des = open (lab + '-intrinsic-moments.txt', 'w+')
    #des.write('----------\nstrain: '+str(mt[0][0])+'\n')
    des.write('\n0th: ' + str(mt[0][1]) + '\n')
    des.write('\n1th: ' + str(mt[0][2][0])+'\t'+str(mt[0][2][1])+'\t'+str(mt[0][2][2])+ '\n')
    des.write('\n2nd: ' + '\n')
    for i1 in range(3):
        for i2 in range(3):
            des.write(str(mt[0][3][i1][i2])+'\t')
        des.write('\n')
    des.write('\n2nd Eigen: ' + str(mt[0][4][0])+'\t'+str(mt[0][4][1])+'\t'+str(mt[0][4][2])+ '\n')
    des.write('\n3rd: ' + '\n')
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                des.write(str(mt[0][5][i1][i2][i3])+'\t')
            des.write('\n')
        des.write('\n')
    des.write('\n4th: ' + '\n')
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    des.write(str(mt[0][6][i1][i2][i3][i4])+'\t')
                des.write('\n')
            des.write('\n')
        des.write('\n')
    des.write('\n5th: ' + '\n')
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        des.write(str(mt[0][7][i1][i2][i3][i4][i5])+'\t')
                    des.write('\n')
                des.write('\n')
            des.write('\n')
        des.write('\n')
    des.close()


if __name__ == "__main__":


    rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/BFS/finalNode/depth5"
    # for linkerDir in os.listdir(rootPath):
    for i in range(6395,110000):
        linkerDir = "linker" + str(i)
        os.chdir(rootPath + "/" + linkerDir + "/" + linkerDir + "_deformation")
        intriMoment(linkerDir)
        momentList = pullMomentforSteps(linkerDir)
        f = open(linkerDir + '-56Moments.txt', 'w')
        for i in range(len(momentList)):
            f.writelines(str(momentList[i]))
            if i != len(momentList) - 1:
                f.writelines(",")
        f.close()

    # time0 = time.time()
    # lab = "linker0"
    # path = "/Users/chengxiyang/PycharmProjects/MORF/calr/testLinkerDir/linker0/linker0_deformation/"
    # os.chdir(path)
    #
    # intriMoment(lab)
    # print(pullMomentforSteps(lab))
    #
    # time1 = time.time()
    # print(time1 - time0)



