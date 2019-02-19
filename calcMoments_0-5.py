
import sys
import os
import numpy as np
import copy
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

def readOneFrame(file, endPoint):
    positions = []
    for line in file:
        if line == endPoint:
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

def strainRead(lab):
    # Read data from file
    res = open(lab + "-ave-force.d", 'r')
    oriData = []
    for line in res:
        words = line.split()
        if len(words)==0:
            continue
        if words[0]!='#':
            tempt = []
            for word in words:
                tempt.append(float(word))
            oriData.append(tempt)

    #Pre-analysis of the raw data
    len0 = oriData[0][2]
    dataReorg = []
    for e in oriData:
        dataReorg.append([e[0],100*(e[2]-len0)/len0])
    return dataReorg

def matchStrain(timestep, map):
    smaller = map[0]
    while len(map) > 0 and map[0][0] < timestep:
        smaller = map[0]
        map.pop(0)
    if map[0][0] == timestep:
        return map[0][1]
    else:
        if len(map) == 0:
            return smaller[1]
        else:
            return (smaller[1]+map[0][1])/2.0

def calcMomentforSteps(lab):
    positionsFile = open(lab + '-deform-nvt.xyz', 'r')
    endPoint = positionsFile.readline()
    mass = readTypeInfo(lab)
    strainInfo = strainRead(lab)
    data = []
    while True:
        try:
            timeInfo = positionsFile.readline().split()
            timeStep = float(timeInfo[2])
            strain = matchStrain(timeStep, strainInfo)
            position = readOneFrame(positionsFile, endPoint)
            zerothMoment, firstMoment, secondMoment, eigenSecondMoment, thirdMoment, fourthMoment, fifthMoment = calcMoment(mass, position)
            data.append([strain, zerothMoment, firstMoment, secondMoment, eigenSecondMoment, thirdMoment, fourthMoment, fifthMoment])

        except:
            break
    mc = []
    mt = []
    for i in range(len(data)):
        if i == 0:
            mc.append(data[i])
            continue
        if (data[i-1][0] - data[i][0]) < 0:
            mt.append(data[i])
        else:
            mc.append(data[i])
    mc = sorted(mc, key=lambda x : x[0])
    mt = sorted(mt, key=lambda x : x[0])

    #Get rid of repeated xvalue:
    i = 0
    while i in range(len(mc) - 1):
        count = 1.0
        newData = mc[i]
        while i < (len(mc) - 1) and mc[i][0] == mc[i + 1][0]:
            count = count + 1
            for j in range(len(mc[i])):
                newData[j] = newData[j] + mc[i + 1][j]
            mc.pop(i)
        for j in range(len(mc[i])):
            newData[j] = newData[j] / count
        mc[i] = newData
        i = i + 1
    des = open(lab + '-tension-moments.txt', 'w+')
    while len(mt) > 0:
        des.write('----------\nstrain: '+str(mt[0][0])+'\n')
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
        mt.pop(0)
    des.close()
    des = open(lab + '-compression-moments.txt', 'w+')
    while len(mc) > 0:
        des.write('----------\nstrain: '+str(mc[0][0])+'\n')
        des.write('\n0th: ' + str(mc[0][1]) + '\n')
        des.write('\n1th: ' + str(mc[0][2][0])+'\t'+str(mc[0][2][1])+'\t'+str(mc[0][2][2])+ '\n')
        des.write('\n2nd: ' + '\n')
        for i1 in range(3):
            for i2 in range(3):
                des.write(str(mc[0][3][i1][i2])+'\t')
            des.write('\n')
        des.write('\n2nd Eigen: ' + str(mc[0][4][0])+'\t'+str(mc[0][4][1])+'\t'+str(mc[0][4][2])+ '\n')
        des.write('\n3rd: ' + '\n')
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    des.write(str(mc[0][5][i1][i2][i3])+'\t')
                des.write('\n')
            des.write('\n')
        des.write('\n4th: ' + '\n')
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    for i4 in range(3):
                        des.write(str(mc[0][6][i1][i2][i3][i4])+'\t')
                    des.write('\n')
                des.write('\n')
            des.write('\n')
        des.write('\n5th: ' + '\n')
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    for i4 in range(3):
                        for i5 in range(3):
                            des.write(str(mc[0][7][i1][i2][i3][i4][i5])+'\t')
                        des.write('\n')
                    des.write('\n')
                des.write('\n')
            des.write('\n')
        mc.pop(0)
    des.close()


#dir = "/Users/weiyi/Documents/Projects/FD-curve/moments-Test/"
#des = open('moment_calc_process.txt', 'w+')
dir = "/rhome/wzhan097/shared/CleanMORF/randomOutput/Trail200/candidate/"
os.chdir(dir)
#trails = os.listdir(os.getcwd())
trails = []
start = int(sys.argv[1])
end = int(sys.argv[2])
for i in range(start,end+1,1):
    trails.append(str(int(i)))
for trail in trails:
    try:
        os.chdir(dir+"/"+trail)
        depths = os.listdir(os.getcwd())
        for depth in depths:
            try:
                os.chdir(dir + "/" + trail + "/" + depth)
                linkers = os.listdir(os.getcwd())
                for linker in linkers:
                    words = linker.split("_")
                    if words[-1] == 'deformation':
                        lab = words[0]
                        os.chdir(dir + "/" + trail + "/" + depth + "/" + linker)
                        calcMomentforSteps(lab)
            except:
                continue
        #des.write(trail)
    except:
        continue
#des.close()
