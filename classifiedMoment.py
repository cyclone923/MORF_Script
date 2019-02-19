# Python 3
# Pull out information from linker{#}-ave-force.d, Calculate the needed results, fit the data to a function and shift
# the curve through zero point



import math
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import scipy.interpolate as inter

import sys
import os
import numpy as np

elements = np.array(['C', 'H', 'O', 'N'])
elementMass = np.array([12, 1, 16, 14])

def transNoise(data):
    return 1.0/(data+1.0)

def SetDirectory(n,folder):
    n = int(n)
    dir = '0-159999'
    os.chdir(folder)
    if dir not in os.listdir(os.getcwd()):
        return False
    os.chdir(folder + dir)
    dir = dir + '/' + str( int(n // 8000 * 8000) ) + '-' + str( int(n // 8000 * 8000 + 7999))
    if str( int(n // 8000 * 8000) ) + '-' + str( int(n // 8000 * 8000 + 7999)) not in os.listdir(os.getcwd()):
        return False
    os.chdir(folder + dir+ '/')
    dir = dir + '/' + str( int(n // 400 * 400) ) + '-' + str( int(n // 400 * 400 + 399) )
    if str( int(n // 400 * 400) ) + '-' + str( int(n // 400 * 400 + 399) ) not in os.listdir(os.getcwd()):
        return False
    os.chdir(folder + dir+ '/')
    dir = dir + '/' + str( int(n // 20 * 20) ) + '-' + str( int(n // 20 * 20 + 19)) + '/'
    if str( int(n // 20 * 20) ) + '-' + str( int(n // 20 * 20 + 19)) not in os.listdir(os.getcwd()):
        return False
    return dir;

def readMom(momentMat, order, *args):
    if order == 0:
        mom = momentMat[order]
    if order == 1:
        mom = momentMat[order][args[0]]
    if order == 2:
        mom = (momentMat[order][args[0]][args[1]])
    if order == 3:
        mom = momentMat[order][args[0]][args[1]][args[2]]
    if order == 4:
        mom = momentMat[order][args[0]][args[1]][args[2]][args[3]]
    if order == 5:
        mom = momentMat[order][args[0]][args[1]][args[2]][args[3]][args[4]]
    if order == 6:
        mom = momentMat[order][args[0]][args[1]][args[2]][args[3]][args[4]][args[5]]
    if order == 7:
        mom = momentMat[order][args[0]][args[1]][args[2]][args[3]][args[4]][args[5]][args[6]]
    if order == 8:
        mom = momentMat[order][args[0]][args[1]][args[2]][args[3]][args[4]][args[5]][args[6]][args[7]]
    if order == 9:
        mom = momentMat[order][args[0]][args[1]][args[2]][args[3]][args[4]][args[5]][args[6]][args[7]][args[8]]
    if order == 10:
        mom = momentMat[order][args[0]][args[1]][args[2]][args[3]][args[4]][args[5]][args[6]][args[7]][args[8]][args[9]]

    if mom == 0:
        return mom
    else:
        if order == 0:
            return mom
        mom = mom/abs(mom) * (abs(mom)**(1/order))
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

    sixthMoment = []
    temp = sixthMoment.copy()
    for order in range(6):
        for i in range(3):
            sixthMoment.append(temp)
        temp = sixthMoment.copy()
        sixthMoment.clear()
    sixthMoment = temp.copy()
    temp.clear()
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        for i6 in range(3):
                            sixthMoment[i1][i2][i3][i4][i5][i6] = 0.0
    sixthMoment = np.array(sixthMoment)
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        for i6 in range(3):
                            sixthMoment[i1][i2][i3][i4][i5][i6] = (sum(
                                [(mass / zerothMoment * position[i1] * position[i2] * position[i3] * position[i4] * position[i5] * position[i6]) for
                                 mass, position in zip(masses, positions)]))

    seventhMoment = []
    temp = seventhMoment.copy()
    for order in range(7):
        for i in range(3):
            seventhMoment.append(temp)
        temp = seventhMoment.copy()
        seventhMoment.clear()
    seventhMoment = temp.copy()
    temp.clear()
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        for i6 in range(3):
                            for i7 in range(3):
                                seventhMoment[i1][i2][i3][i4][i5][i6][i7] = 0.0
    seventhMoment = np.array(seventhMoment)
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        for i6 in range(3):
                            for i7 in range(3):
                                seventhMoment[i1][i2][i3][i4][i5][i6][i7] = (sum(
                                    [(mass / zerothMoment * position[i1] * position[i2] * position[i3] * position[i4] * position[i5] * position[i6] * position[i7]) for
                                    mass, position in zip(masses, positions)]))

    eighthMoment = []
    temp = eighthMoment.copy()
    for order in range(8):
        for i in range(3):
            eighthMoment.append(temp)
        temp = eighthMoment.copy()
        eighthMoment.clear()
    eighthMoment = temp.copy()
    temp.clear()
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        for i6 in range(3):
                            for i7 in range(3):
                                for i8 in range(3):
                                    eighthMoment[i1][i2][i3][i4][i5][i6][i7][i8] = 0.0
    eighthMoment = np.array(eighthMoment)
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        for i6 in range(3):
                            for i7 in range(3):
                                for i8 in range(3):
                                    eighthMoment[i1][i2][i3][i4][i5][i6][i7][i8] = (sum(
                                        [(mass / zerothMoment * position[i1] * position[i2] * position[i3] * position[i4] * position[i5] * position[i6] * position[i7] * position[i8]) for
                                        mass, position in zip(masses, positions)]))

    ninthMoment = []
    temp = ninthMoment.copy()
    for order in range(9):
        for i in range(3):
            ninthMoment.append(temp)
        temp = ninthMoment.copy()
        ninthMoment.clear()
    ninthMoment = temp.copy()
    temp.clear()
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        for i6 in range(3):
                            for i7 in range(3):
                                for i8 in range(3):
                                    for i9 in range(3):
                                        ninthMoment[i1][i2][i3][i4][i5][i6][i7][i8][i9] = 0.0
    ninthMoment = np.array(ninthMoment)
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        for i6 in range(3):
                            for i7 in range(3):
                                for i8 in range(3):
                                    for i9 in range(3):
                                        ninthMoment[i1][i2][i3][i4][i5][i6][i7][i8][i9] = (sum(
                                            [(mass / zerothMoment * position[i1] * position[i2] * position[i3] * position[i4] * position[i5] * position[i6] * position[i7] * position[i8] * position[i9]) for
                                            mass, position in zip(masses, positions)]))

    tenthMoment = []
    temp = tenthMoment.copy()
    for order in range(10):
        for i in range(3):
            tenthMoment.append(temp)
        temp = tenthMoment.copy()
        tenthMoment.clear()
    tenthMoment = temp.copy()
    temp.clear()
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        for i6 in range(3):
                            for i7 in range(3):
                                for i8 in range(3):
                                    for i9 in range(3):
                                        for i10 in range(3):
                                            tenthMoment[i1][i2][i3][i4][i5][i6][i7][i8][i9][i10] = 0.0
    tenthMoment = np.array(tenthMoment)
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        for i6 in range(3):
                            for i7 in range(3):
                                for i8 in range(3):
                                    for i9 in range(3):
                                        for i10 in range(3):
                                            tenthMoment[i1][i2][i3][i4][i5][i6][i7][i8][i9][i10] = (sum(
                                                [(mass / zerothMoment * position[i1] * position[i2] * position[i3] * position[i4] * position[i5] * position[i6] * position[i7] * position[i8] * position[i9] * position[i10]) for mass, position in zip(masses, positions)]))

    return [zerothMoment, firstMoment, secondMoment, thirdMoment, fourthMoment, fifthMoment, sixthMoment, seventhMoment, eighthMoment, ninthMoment, tenthMoment]

def intriMoment(lab):
    positionsFile = open(lab + '-un-squashed.xyz', 'r')
    positionsFile.readline()
    positionsFile.readline()
    mass = readTypeInfo(lab)
    position = readOneFrame(positionsFile)
    totMass = sum(np.array(mass))
    massCenter = calcMassCenter(mass,position)
    transMatrix = calcNewCoor(position)
    for i in range(len(position)):
        position[i] = newPosition(position[i], transMatrix, massCenter)
    classifiedMasses = []
    classifiedPositions = []
    for i in range(len(mass)):
        if mass[i] not in classifiedMasses:
            classifiedMasses.append(mass[i])
            classifiedPositions.append([position[i]])
        else:
            classifiedPositions[classifiedMasses.index(mass[i])].append(position[i])
    classifiedMoments = []
    for i in range(len(classifiedMasses)):
        classifiedMoments.append(calcMoment([classifiedMasses[i]]*len(classifiedPositions[i]), classifiedPositions[i]))
    #zerothMoment, firstMoment, secondMoment, eigenSecondMoment, thirdMoment, fourthMoment, fifthMoment = calcMoment(mass, position)
    #moments = []
    #moments.append(zerothMoment)
    #moments.append(firstMoment)
    #moments.append(secondMoment)
    #moments.append(thirdMoment)
    #moments.append(fourthMoment)
    #moments.append(fifthMoment)
    positionsFile.close()
    return classifiedMasses, classifiedMoments

def independentMoments(momentsMat):
    moments = []
    for i in range(11):
        moments.append([])
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                for i4 in range(3):
                    for i5 in range(3):
                        for i6 in range(3):
                            for i7 in range(3):
                                for i8 in range(3):
                                    for i9 in range(3):

                                        for i10 in range(3):
                                            temp = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10]
                                            temp.sort()
                                            if temp not in moments[10]:
                                                moments[10].append(temp)

                                        temp = [i1, i2, i3, i4, i5, i6, i7, i8, i9]
                                        temp.sort()
                                        if temp not in moments[9]:
                                            moments[9].append(temp)
                                    temp = [i1, i2, i3, i4, i5, i6, i7, i8]
                                    temp.sort()
                                    if temp not in moments[8]:
                                        moments[8].append(temp)
                                temp = [i1, i2, i3, i4, i5, i6, i7]
                                temp.sort()
                                if temp not in moments[7]:
                                    moments[7].append(temp)
                            temp = [i1, i2, i3, i4, i5, i6]
                            temp.sort()
                            if temp not in moments[6]:
                                moments[6].append(temp)
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
    #print (moments)
    data = []
    for i in range(11):
        for e in moments[i]:
            data.append(readMom(momentsMat, i, *e))
    return data

def calcFileMoment(lab):
    masses, moments = intriMoment(lab)
    for i in range(len(masses)):
        moments[i] = independentMoments(moments[i])
    return masses, moments

def singleTask(lab):
    if lab + "-ave-force.d" not in os.listdir(os.getcwd()) or lab + '-un-squashed.xyz' not in os.listdir(os.getcwd()):
        return False
    masses, moments = calcFileMoment(lab)
    return masses, moments

if __name__ == "__main__":
    # d = sys.argv[1]
    start = sys.argv[1]
    end = sys.argv[2]
    rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/BFS/finalNode/depth5"  #d4 302-6394 d5 6395-171390
    # for linkerDir in os.listdir(rootPath):
    for i in range(int(start),int(end)):
        linkerDir = "linker" + str(i)
        os.chdir(rootPath + "/" + linkerDir + "/" + linkerDir + "_deformation")
        masses, moments= singleTask(linkerDir)
        fileElements = []
        momentArray = np.zeros(shape=(4,286))
        for e in masses:
            fileElements.append(np.where(elementMass == round(e))[0][0])
        for j in range(len(elements)):
            if j in fileElements:
                elementNo = fileElements.index(j)
                momentArray[j] = moments[elementNo]
            else:
                momentArray[j] = 0

        # print(momentArray.reshape(-1).shape)
        np.save(linkerDir + '-classified-moments.npy', momentArray)
    print("Done")


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
    #                 masses, moments= singleTask(linkerDir)
    #                 fileElements = []
    #                 momentArray = np.zeros(shape=(4,286))
    #                 for e in masses:
    #                     fileElements.append(np.where(elementMass == round(e))[0][0])
    #                 for j in range(len(elements)):
    #                     if j in fileElements:
    #                         elementNo = fileElements.index(j)
    #                         momentArray[j] = moments[elementNo]
    #                     else:
    #                         momentArray[j] = 0

    #                 print(momentArray.reshape(-1).shape)
    #                 np.save(linkerDir + '-classified-moments.npy', momentArray.reshape(-1))