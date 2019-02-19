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

orders = ['0th','1th','2nd','3rd','4th','5th']


def transAve(data):
    if abs(data) > 50:
        data = abs(data)/data*50
    if data == 0:
        return 1
    return 1.0/(data+abs(data)/data)

def transJump(data):
    if abs(data) < 1:
        return 1.0
    if abs(data) > 50:
        data = abs(data)/data*50
    return 1+1.0/(data)

def transDiff(data):
    if data == 0:
        return 1.0
    return 1.0/(data+1)

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
    moments = calcMoment(mass, position)
    #zerothMoment, firstMoment, secondMoment, eigenSecondMoment, thirdMoment, fourthMoment, fifthMoment = calcMoment(mass, position)
    #moments = []
    #moments.append(zerothMoment)
    #moments.append(firstMoment)
    #moments.append(secondMoment)
    #moments.append(thirdMoment)
    #moments.append(fourthMoment)
    #moments.append(fifthMoment)
    positionsFile.close()
    return moments

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

def bondMom(lab):
    atom = []
    bond = []
    stiffness = []
    #atomFile = open(lab+'.coeff', 'r')
    atomFile = open(lab+'.lmpdat', 'r')
    for line in atomFile:
        if line == 'Atoms\n':
            break
    atomFile.readline()
    line = atomFile.readline()
    while line != '\n':
        info = line.split()
        atom.append([float(info[-3]), float(info[-2]), float(info[-1])])
        line = atomFile.readline()
    for line in atomFile:
        if line == 'Bonds\n':
            break
    atomFile.readline()
    line = atomFile.readline()
    while line != '\n':
        info = line.split()
        bond.append([[int(info[-2]), int(info[-1])], int(info[1])])
        line = atomFile.readline()
    atomFile.close()

    bondFile = open(lab+'.coeff', 'r')
    #bondFile = open(lab+'.lmpdat', 'r')
    for line in bondFile:
        if line!='\n':
            info = line.split()
            if info[0]=='bond_coeff':
                stiffness.append(float(info[3]))
    bondFile.close()

    for e in bond:
        e[0][0] = atom[e[0][0] - 1]
        e[0][1] = atom[e[0][1] - 1]
        e[0] = [(x + y) / 2.0 for x, y in zip(e[0][0], e[0][1])]
        e[1] = stiffness[e[1] - 1] / 2.0

    result = calcMoment([x[1] for x in bond], [x[0] for x in bond])
    result = (independentMoments(result))

    return result

def calcFileMoment(lab):
    moments = intriMoment(lab)
    moments = independentMoments(moments)
    return moments

def findMin(m, _function_):
    """
    Return the minimum value of the function within the range of the m matrix covered
    :param m: input data matrix
    :param _function_: functionname
    :return: minimum of curve and the corresponding variable vlalue
    """
    m = np.array(m)
    _from = min(m[:,0])
    _to = max(m[:,0])
    step = (_to - _from)/10000
    knots = [x*step + _from for x in range(10000+1)]
    minNo=_function_(knots[0])
    minXNo=knots[0]
#    print(knots)
    yvalue = []
    for e in knots:
        yvalue.append(_function_(e))
        if yvalue[-1] < minNo:
            minNo = yvalue[-1]
            minXNo = e
    return minXNo,minNo

def derivation(func, values, delta):
    return (func(values+delta)-func(values))/delta

def findBestMatch(x, y):
    """
    This function is designed to find the best s value for spline
    :param x:
    :param y:
    :return: the min and max of best s range
            if the spline cannot be smooth enough even with s = 5000(maxS), minS will be returned as False
    """
    threshold = 50
    def findMatchRange(x, y, step, start, end):
        trailList = np.linspace(start, end, (end-start)/step + 1)
        _from = min(x)
        _to = max(x)
        testPointsX = np.array([e * (_to - _from) / 500 + _from for e in range(500 + 1)])
        for trail in trailList:
            tempFunc = inter.UnivariateSpline (x, y, s = trail)
            if max(abs(derivation(tempFunc, testPointsX, 0.0001))) < threshold:
                if trail == 0:
                    return trail, trail+step
                else:
                    return trail-step, trail
        if threshold == 50:
            return 'False', trail
        return trail-step, trail
    minS = 0
    maxS = 5000
    while maxS-minS > 5:
        delta = (maxS-minS)/32.0
        minS, maxS = findMatchRange(x, y, delta, minS, maxS)
        if minS == 'False':
            return minS, maxS
    while minS < 100 and threshold > 2:
        threshold = threshold / 2.0
        minS = 0
        maxS = 200
        while maxS - minS > 5:
            delta = (maxS - minS) / 32.0
            minS, maxS = findMatchRange(x, y, delta, minS, maxS)
    return minS, maxS

def calcCurveFeatures(mc, mt):
    '''
    Get the FD and PE information from the raw data of compression and tension process
    Polyfit the FD curve and get the flexibility of the molecule woth given label
    Pull out the features of the compression and tension procedure
    Plot out the relative curves and return feature values
    :param mc: compression data
    :param mt: tension data
    :return: _ifYesJumpC_, _ifYesJumpT_, _diffOfCT_, _diffOfTC_, _aveCB_, _aveCF_, _aveTB_, _aveTF_
        _ifYesJumpC_:	The Jump Value* of Compression Curve (If no Jump = 0)
            [like in the second group, the abnormal condition having the jump value]
        _ifYesJumpC_:	The Jump Value of Tension Curve (If no Jump = 0)
            [like in the second group, the abnormal condition having the jump value]
        _diffOfCT_:	The largest value of (Force of Compression - Force of Tension) at the same strain
        _diffOfCT_:	The largest value of (Force of Tension - Force of Compression) at the same strain
        _aveCB_:	The high flexibility of Compression*
        _aveCF_:	The low flexibility of Compression
        _aveTB_:	The high flexibility of Tension
        _aveTF_:	The low flexibility of Tension
        *The Jump value is the largest abnormal flexibility*
        *If there is no obvious flexibility changing _ave*B_ will equal _ave*F_ *
    '''

    mcn = np.array(mc)
    mtn = np.array(mt)

    #Polyfit the compression and tension curve with degree 5
    coeffs = poly.polyfit(mcn[:,0],mcn[:,-1],5)
    pfit = poly.Polynomial(coeffs)

    m = np.array(list(map(lambda x,y:[x,y],mcn[:,0],mcn[:,-1])))
    # Shift the curve to satisfy the rule of lowest energy as 0 and at 0% strain
    eo, PEo = findMin(m, pfit)

#        return "Error: PE min outside range"
    mcn[:,0] -= eo
    mtn[:,0] -= eo
    mcn[:,-1] -= PEo
    mtn[:,-1] -= PEo
    m = np.array(list(map(lambda x, y: [x, y], mcn[:, 0], mcn[:, -1])))

    # Get the spline function of FD curve for further evaluation and plot the fitted curve

    minSC, maxS = findBestMatch(mcn[:, 0], mcn[:, 2])
    fdFunctionC = inter.UnivariateSpline(mcn[:, 0], mcn[:, 2], s = maxS)
    if minSC == 'False':
        minSC = maxS
    fdNoiseC = maxS
    minST, maxS = findBestMatch(mtn[:, 0], mtn[:, 2])
    fdFunctionT = inter.UnivariateSpline(mtn[:, 0], mtn[:, 2], s = maxS)
    if minST == 'False':
        minST = maxS
    fdNoiseT = maxS

    # Calculate the flexibility of the molecule with 'flexibility = PresentLength*Delta(Force)/Delta(Length)'
    # plot the flexibility information and save the data
    flexibilityC = []
    flexibilityT = []
    _from = min(m[:,0])
    _to = max(m[:,0])
    step = (_to - _from)/500
    xPoints = [x*step + _from for x in range(500+1)]
    deltaX = 0.0001

    # Get the differences of tension and compression process and evaluate the jump abnormal situation
    _ifYesJumpC_ = 0
    _ifYesJumpT_ = 0
    _diffOfCT_ = 0
    _diffOfTC_ = 0
    for e in xPoints:
        flexibilityC.append([e,(fdFunctionC(e+deltaX)-fdFunctionC(e))/deltaX * ((100+e)/100.0)])
        flexibilityT.append([e,(fdFunctionT(e+deltaX)-fdFunctionT(e))/deltaX * ((100+e)/100.0)])
        #flexibilityC.append([e,(abs(fdFunctionC(e+deltaX)-fdFunctionC(e)))/deltaX * ((100+e)/100.0)])
        #flexibilityT.append([e,(abs(fdFunctionT(e+deltaX)-fdFunctionT(e)))/deltaX * ((100+e)/100.0)])

        if flexibilityC[-1][1] < -10 and flexibilityC[-1][1] < _ifYesJumpC_:   #The Threshold was set randomly, in need of more trials.
            _ifYesJumpC_ = flexibilityC[-1][1]
        if flexibilityT[-1][1] < -10 and flexibilityT[-1][1] < _ifYesJumpT_:  # The Threshold was set randomly, in need of more trials.
            _ifYesJumpT_ = flexibilityT[-1][1]
        #flexibilityC[-1][1] = abs(flexibilityC[-1][1])
        #flexibilityT[-1][1] = abs(flexibilityT[-1][1])

        if fdFunctionC(e) - fdFunctionT(e) > _diffOfCT_:
            _diffOfCT_ = fdFunctionC(e) - fdFunctionT(e)
        if fdFunctionT(e) - fdFunctionC(e) > _diffOfTC_:
            _diffOfTC_ = fdFunctionT(e) - fdFunctionC(e)


    # Calculate the average of flexibility before and after bulk

    _aveCF_ = flexibilityC[0][1]
    _aveTF_ = flexibilityT[0][1]
    _aveCB_ = flexibilityC[-1][1]
    _aveTB_ = flexibilityT[-1][1]
    _countCF_ = 1
    _countTF_ = 1
    _countCB_ = 1
    _countTB_ = 1
    aveThresholdC = 10
    aveThresholdT = 10
    if minSC < 250:
        aveThresholdC = 30
    if minST < 250:
        aveThresholdT = 30
    for i in range(len(flexibilityC) - 1):
        if _countCF_<  100 and (_countCF_ < aveThresholdC or abs(flexibilityC[i + 1][1] - (_aveCF_/_countCF_)) < 3.5 + fdNoiseC/1000.0):    #The Threshold was set randomly, in need of more trials.
            _aveCF_ += flexibilityC[i + 1][1]
            _countCF_ += 1
        if _countCB_<100 and (_countCB_ < aveThresholdC or abs(flexibilityC[-i - 2][1] - (_aveCB_/_countCB_)) < 3.5 + fdNoiseC/1000.0):    #The Threshold was set randomly, in need of more trials.
            _aveCB_ += flexibilityC[-i - 2][1]
            _countCB_ += 1
        if _countTF_<100 and (_countTF_ < aveThresholdT or abs(flexibilityT[i + 1][1] - (_aveTF_/_countTF_)) < 3.5 + fdNoiseT/1000.0):    #The Threshold was set randomly, in need of more trials.
            _aveTF_ += flexibilityT[i + 1][1]
            _countTF_ += 1
        if _countTB_<100 and (_countTB_ < aveThresholdT or abs(flexibilityT[-i - 2][1] - (_aveTB_/_countTB_)) < 3.5 + fdNoiseT/1000.0):    #The Threshold was set randomly, in need of more trials.
            _aveTB_ += flexibilityT[-i - 2][1]
            _countTB_ += 1
    _aveCB_ = _aveCB_ / _countCB_
    _aveCF_ = _aveCF_ / _countCF_
    _aveTB_ = _aveTB_ / _countTB_
    _aveTF_ = _aveTF_ / _countTF_

    if abs(_aveCB_- flexibilityC[-1][1]) > fdNoiseC/100.0:    #The Threshold was set randomly, in need of more trials.
        i=0
        while abs(_aveCB_- flexibilityC[-(1+i)][1]) > fdNoiseC/100.0 and i < _countCB_ - 5:
            _aveCB_ = (_aveCB_ * (_countCB_ - i) - flexibilityC[-(1+i)][1])/(_countCB_-1-i)
            i = i+1
    if abs(_aveCF_- flexibilityC[0][1]) > fdNoiseC/100.0:    #The Threshold was set randomly, in need of more trials.
        i=0
        while abs(_aveCF_- flexibilityC[0+i][1]) > fdNoiseC/100.0 and i < _countCF_ - 5:
            _aveCF_ = (_aveCF_ * (_countCF_ - i) - flexibilityC[-1+i][1])/(_countCF_-1-i)
            i = i + 1
    if abs(_aveTB_- flexibilityT[-1][1]) > fdNoiseT/100.0:    #The Threshold was set randomly, in need of more trials.
        i=0
        while abs(_aveTB_- flexibilityT[-(1+i)][1]) > fdNoiseT/100.0 and i < _countTB_ - 5:
            _aveTB_ = (_aveTB_ * (_countTB_ - i) - flexibilityT[-(1+i)][1])/(_countTB_-1-i)
            i = i + 1
    if abs(_aveTF_- flexibilityT[0][1]) > fdNoiseT/100.0:    #The Threshold was set randomly, in need of more trials.
        i = 0
        while abs(_aveTF_ - flexibilityT[0 + i][1]) > fdNoiseT / 100.0 and i < _countTF_ - 5:
            _aveTF_ = (_aveTF_ * (_countTF_ - i) - flexibilityT[-1 + i][1]) / (_countTF_ - 1 - i)
            i = i + 1

    return [_ifYesJumpC_, _ifYesJumpT_, _diffOfCT_, _diffOfTC_, _aveCB_, _aveCF_, _aveTB_, _aveTF_, fdNoiseC, fdNoiseT]

#curve_fit

def calcFileFeatures(res):
    '''
    Read data from the input file and finish the preparation
    Quoting the 'createFit' function to analyze the data
    :param res: input file
    :return: The feature of the input data which is the return value of quoted function
    '''

    # Read data from file
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
    l = len(oriData)
    nav = 5
    giveUpNo =l-l//nav*nav
    if giveUpNo != 0:
        oriData = oriData[:-(giveUpNo)]
    len0 = oriData[0][2]
    dataReorg = []
    for e in oriData:
        f = [0.5 * (e[3] - e[6]), 0.5 * (e[4] - e[7]), 0.5 * (e[5] - e[8])]
        v = e[9:12]
        mod = (v[0]**2+v[1]**2+v[2]**2)**0.5
        d = [x / mod for x in v]
        fax = f[0]*d[0]+f[1]*d[1]+f[2]*d[2]
        fn = [f[0]-[x * fax for x in d][0],f[1]-[x * fax for x in d][1],f[2]-[x * fax for x in d][2]]
        mod = (fn[0]**2+fn[1]**2+fn[2]**2)**0.5
        dataReorg.append([100*(e[2]-len0)/len0,e[2]-len0,fax,mod,e[12]])
    data = []
    for i in range(int(len(dataReorg))):
        if i/nav == int(i/nav):
            data.append([])
            for j in range(nav):
                data[i//nav].append(0)
                if i//nav != 0:
                    data[i // nav - 1][j] = data[i // nav - 1][j]/nav
        for j in range(nav):
            data[i//nav][j]+=dataReorg[i][j]

    for j in range(nav):
        data[-1][j] = data[-1][j]/nav

    # Seperate the raw data to compression and tension process
    mc = []
    mt = []
    for i in range(len(data)):
        if i == 0:
            mc.append(data[i])
            continue
        if (data[i-1][1] - data[i][1]) < 0:
            mt.append(data[i])
        else:
            mc.append(data[i])
    mc = sorted(mc, key=lambda x : x[0])
    mt = sorted(mt, key=lambda x: x[0])

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
    i = 0
    while i in range(len(mt) - 1):
        count = 1.0
        newData = mt[i]
        while i < (len(mt) - 1) and mt[i][0] == mt[i + 1][0]:
            count = count + 1
            for j in range(len(mt[i])):
                newData[j] = newData[j] + mt[i + 1][j]
            mt.pop(i)
        for j in range(len(mt[i])):
            newData[j] = newData[j] / count
        mt[i] = newData
        i = i + 1


    return calcCurveFeatures(mc, mt)

def transFeature(_ifYesJumpC_, _ifYesJumpT_, _diffOfCT_, _diffOfTC_, _aveCB_, _aveCF_, _aveTB_, _aveTF_, _fdNoiseC_, _fdNoiseT_):
    data = []
    data.append((transAve(_aveCB_))** 1)
    #print(reward)
    data.append((transAve(_aveTB_))** 1)
    #print(reward)
    data.append((transAve(_aveCF_))** 1)
    #print(reward)
    data.append((transAve(_aveTF_))** 1)
    #print(reward)
    data.append((transJump(_ifYesJumpC_))** 1)
    #print(reward)
    data.append((transJump(_ifYesJumpT_))** 1)
    #print(reward)
    data.append((transDiff(_diffOfCT_))** 1)
    #print(reward)
    data.append((transDiff(_diffOfTC_))** 1)
    #print(reward)
    data.append((transNoise(_fdNoiseC_))** 1)
    #print(reward)
    data.append((transNoise(_fdNoiseT_))** 1)
    #print(reward)
    return data

def singleTask(lab):
    if lab + "-ave-force.d" not in os.listdir(os.getcwd()) or lab + '-un-squashed.xyz' not in os.listdir(os.getcwd()):
        return False
    res = open(lab + "-ave-force.d","r")
    features = calcFileFeatures(res)
    res.close()
    transFeatures = transFeature(*features)
    moments = calcFileMoment(lab)
    bondMoments = bondMom(lab)
    return bondMoments, moments, features, transFeatures

if __name__ == "__main__":
    # d = sys.argv[1]
    start = sys.argv[1]
    end = sys.argv[2]
    rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/BFS/finalNode/depth5" #d4 302-6394 d5 6395-110000
    # for linkerDir in os.listdir(rootPath):
    for i in range(int(start),int(end)):
        linkerDir = "linker" + str(i)
        os.chdir(rootPath + "/" + linkerDir + "/" + linkerDir + "_deformation")
        lab = linkerDir
        bondMomentList, momentList, featuresList, transFeatureList = singleTask(lab)
        f = open(lab + '-bondMoment.txt', 'w')
        for i in range(len(bondMomentList)):
            f.writelines(str(bondMomentList[i]))
            if i != len(bondMomentList) - 1:
                f.writelines(",")
        f.close()
        f = open(lab + '-0_10thMoments.txt', 'w')
        for i in range(len(momentList)):
            f.writelines(str(momentList[i]))
            if i != len(momentList) - 1:
                f.writelines(",")
        f.close()
        f = open(lab + '-original-features.txt', 'w')
        for i in range(len(featuresList)):
            f.writelines(str(featuresList[i]))
            if i != len(featuresList) - 1:
                f.writelines(",")
        f.close()
        f = open(lab + '-modified-features.txt', 'w')
        for i in range(len(transFeatureList)):
            f.writelines(str(transFeatureList[i]))
            if i != len(transFeatureList) - 1:
                f.writelines(",")
        f.close()

    # rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/candidate"
    # # start = int(sys.argv[1])
    # # end = int(sys.argv[2])
    #
    # for t in range(0,500):
    #     for d in range(1,31):
    #         depthDir = rootPath + "/" + str(t) + "/Depth" + str(d)
    #         for dir in os.listdir(depthDir):
    #             if dir[-11:] == "deformation":
    #                 # print(t)
    #                 # print(d)
    #                 # print(dir)
    #
    #                 os.chdir(depthDir + "/" + dir)
    #                 lab = dir[:-12]
    #                 bondMomentList, momentList, featuresList, transFeatureList = singleTask(lab)
    #                 f = open(lab + '-bondMoment.txt', 'w')
    #                 for i in range(len(bondMomentList)):
    #                     f.writelines(str(bondMomentList[i]))
    #                     if i != len(bondMomentList) - 1:
    #                         f.writelines(",")
    #                 f.close()
    #                 f = open(lab + '-0_10thMoments.txt', 'w')
    #                 for i in range(len(momentList)):
    #                     f.writelines(str(momentList[i]))
    #                     if i != len(momentList) - 1:
    #                         f.writelines(",")
    #                 f.close()
    #                 f = open(lab + '-original-features.txt', 'w')
    #                 for i in range(len(featuresList)):
    #                     f.writelines(str(featuresList[i]))
    #                     if i != len(featuresList) - 1:
    #                         f.writelines(",")
    #                 f.close()
    #                 f = open(lab + '-modified-features.txt', 'w')
    #                 for i in range(len(transFeatureList)):
    #                     f.writelines(str(transFeatureList[i]))
    #                     if i != len(transFeatureList) - 1:
    #                         f.writelines(",")
    #                 f.close()

