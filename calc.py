from __future__ import division
from math import sqrt
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import scipy.interpolate as inter
import heapq




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


def createFits(mc, mt, lab):
    '''
    Get the FD and PE information from the raw data of compression and tension process
    Polyfit the FD curve and get the flexibility of the molecule woth given label
    Pull out the features of the compression and tension procedure
    Plot out the relative curves and return feature values
    :param mc: compression data
    :param mt: tension data
    :param lab: label of the relative linker
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
    if abs(eo) > 4:
        os.system("touch " + lab + "-min-out-of-range")
#        return "Error: PE min outside range"
    mcn[:,0] -= eo
    mtn[:,0] -= eo
    mcn[:,-1] -= PEo
    mtn[:,-1] -= PEo

    # Plot FD and PE curve after shifted and save the shifted data points
    des = open(lab+"-FD-compression-polyfit.d","w+")
    for e in mcn:
        des.write(str(e[0])+"\t"+str(e[2])+"\t"+str(e[3])+"\t"+str(e[4])+"\n")
    des.close()
    des = open(lab + "-FD-tension-polyfit.d","w+")
    for e in mtn:
        des.write(str(e[0])+"\t"+str(e[2])+"\t"+str(e[3])+"\t"+str(e[4])+"\n")
    des.close()
    plt.figure(figsize=(8.5, 12))
    plt.subplot(211)
    plt.plot(mcn[:,0], mcn[:,2], label='Normal Force - Compression', linewidth=0.5)
    plt.plot(mcn[:,0], mcn[:,1], label='Axial Force - Compression', linewidth=0.5)
    plt.plot(mtn[:,0], mtn[:,2], label='Normal Force - Tension', linewidth=0.5)
    plt.plot(mtn[:,0], mtn[:,1], label='Axial Force - Tension', linewidth=0.5)
    plt.legend()
    plt.xlabel('Strain (%)', family='DejaVu Sans', fontsize=10, color='black')
    plt.ylabel('Force (Kcal/mol/A)', family='DejaVu Sans', fontsize=10, color='black')
    plt.xticks(family='DejaVu Sans', fontsize=10, color='black')
    plt.yticks(family='DejaVu Sans', fontsize=10, color='black')
    plt.subplot(212)
    plt.plot(mcn[:,0], mcn[:,-1], label='Potential Energy - Compression', linewidth=0.5)
    plt.plot(mcn[:,0], mcn[:,-1], label='Potential Energy - Tension', linewidth=0.5)
    plt.legend()
    plt.xlabel('Strain (%)', family='DejaVu Sans', fontsize=10, color='black')
    plt.ylabel('Potential Energy (Kcal/mol)', family='DejaVu Sans', fontsize=10, color='black')
    plt.xticks(family='DejaVu Sans', fontsize=10, color='black')
    plt.yticks(family='DejaVu Sans', fontsize=10, color='black')
    plt.title('')
    plt.savefig(lab + "-FD-PE-curve.jpeg")
    plt.close()

    # Get the spline function of FD curve for further evaluation and plot the fitted curve
    fdFunctionC = inter.UnivariateSpline (mcn[:, 0], mcn[:, 2], s = 200)
    fdFunctionT = inter.UnivariateSpline(mtn[:, 0], mtn[:, 2], s = 200)
    plt.figure()
    plt.plot(mcn[:,0], fdFunctionC(mcn[:,0]), label='Normal Force - Compression', linewidth=0.5)
    plt.plot(mtn[:,0], fdFunctionT(mtn[:,0]), label='Normal Force - Tensi', linewidth=0.5)
    plt.savefig(lab + "-fitting.jpeg")
    plt.close()

    # Calculate the flexibility of the molecule with 'flexibility = PresentLength*Delta(Force)/Delta(Length)'
    # plot the flexibility information and save the data
    flexibilityC = []
    flexibilityT = []
    des = open(lab + "-FD-Flexibility.d","w+")
    des.write("strain\tcompression\ttension\n")
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

        if flexibilityC[-1][1] < -10 and flexibilityC[-1][1] < _ifYesJumpC_:   #The Threshold was set randomly, in need of more trials.
            _ifYesJumpC_ = flexibilityC[-1][1]
        if flexibilityT[-1][1] < -10 and flexibilityT[-1][1] < _ifYesJumpT_:  # The Threshold was set randomly, in need of more trials.
            _ifYesJumpT_ = flexibilityT[-1][1]
        if fdFunctionC(e) - fdFunctionT(e) > _diffOfCT_:
            _diffOfCT_ = fdFunctionC(e) - fdFunctionT(e)
        if fdFunctionT(e) - fdFunctionC(e) > _diffOfTC_:
            _diffOfTC_ = fdFunctionT(e) - fdFunctionC(e)

        des.write(str(e) + '\t' + str(flexibilityC[-1][1]) + '\t' + str(flexibilityT[-1][1]) + '\n')

    # Calculate the average of flexibility before and after bulk
    _flexibilityValueC_ = []
    _flexibilityValueT_ = []
    _aveCF_ = flexibilityC[0][1]
    _aveTF_ = flexibilityT[0][1]
    _aveCB_ = flexibilityC[-1][1]
    _aveTB_ = flexibilityT[-1][1]
    _countCF_ = 1
    _countTF_ = 1
    _countCB_ = 1
    _countTB_ = 1
    for i  in range(len(flexibilityC) - 1):
        if abs(flexibilityC[i + 1][1] - (_aveCF_/_countCF_)) < 3.5:    #The Threshold was set randomly, in need of more trials.
            _aveCF_ += flexibilityC[i + 1][1]
            _countCF_ += 1
        if abs(flexibilityC[-i - 2][1] - (_aveCB_/_countCB_)) < 3.5:    #The Threshold was set randomly, in need of more trials.
            _aveCB_ += flexibilityC[-i - 2][1]
            _countCB_ += 1
        if abs(flexibilityT[i + 1][1] - (_aveTF_/_countTF_)) < 3.5:    #The Threshold was set randomly, in need of more trials.
            _aveTF_ += flexibilityT[i + 1][1]
            _countTF_ += 1
        if abs(flexibilityT[-i - 2][1] - (_aveTB_/_countTB_)) < 3.5:    #The Threshold was set randomly, in need of more trials.
            _aveTB_ += flexibilityT[-i - 2][1]
            _countTB_ += 1
    _aveCB_ = _aveCB_ / _countCB_
    _aveCF_ = _aveCF_ / _countCF_
    _aveTB_ = _aveTB_ / _countTB_
    _aveTF_ = _aveTF_ / _countTF_

    flexibilityT = np.array(flexibilityT)
    flexibilityC = np.array(flexibilityC)

    plt.figure(figsize=(8.5, 5))
    plt.plot(flexibilityC[:,0], flexibilityC[:,1], label='Flexibility - Compression', linewidth=0.5)
    plt.plot(flexibilityT[:, 0], flexibilityT[:, 1], label='Flexibility - Tension', linewidth=0.5)
    plt.legend()
    plt.xlabel('Strain (%)', family='DejaVu Sans', fontsize=10, color='black')
    plt.ylabel('Flexibility (Kcal/mol/A)', family='DejaVu Sans', fontsize=10, color='black')
    plt.xticks(family='DejaVu Sans', fontsize=10, color='black')
    plt.yticks(family='DejaVu Sans', fontsize=10, color='black')
    plt.savefig(lab + "-Flexibility-curve.jpeg")
    plt.close()

    f = open(lab + "-features.txt", "w")
    for i in (_ifYesJumpC_, _ifYesJumpT_, _diffOfCT_, _diffOfTC_, _aveCB_, _aveCF_, _aveTB_, _aveTF_):
        f.write(str(i) + " ")
    f.close()

    return [_ifYesJumpC_, _ifYesJumpT_, _diffOfCT_, _diffOfTC_, _aveCB_, _aveCF_, _aveTB_, _aveTF_]

#curve_fit


def createFile(res, lab):
    '''
    Read data from the input file and finish the preparation
    Quoting the 'createFit' function to analyze the data
    :param res: input file
    :param lab: label of the relative linker
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

    mc = sorted(mc, key=lambda x: x[0])

    # print("--------MC Before-------")
    # for i in mc:
    #     print(i)

    i = 0
    while i < len(mc) - 1:
        if mc[i][0] == mc[i+1][0]:
            j = 1
            if len(mc[i]) != len(mc[j]):
                print("Error, list do not have the same length!")
                exit(0)
            while j < len(mc[i]):
                mc[i][j] = 0.5 * (mc[i][j] + mc[i+1][j])
                j += 1
            mc.pop(i+1)
        i += 1


    # print("--------MC After-------")
    # for i in mc:
    #     print(i)
    # mt = sorted(mt, key=lambda x: x[0])

    return createFits(mc, mt, lab)


def getName(lab):
    filename = lab + "-ave-force.d"
    return filename;

def singleEvaluationTask(lab):
    '''
    Evaluate the data of the linker with the given linker
    *The local directory should be in the deformation folder with the given lab
    :param lab: label of the linker
    :return:
        Return False and the label and the reason for the failure,
            while the required data file is not existing in the directory
        Return False and the label and the reason for the failure,
            while the evaluation is done before
            [represent with the existence of ${linkerlable}-FD-PE-curve.jpeg]
        Return True, the label and the feature gotten by quoting 'createFile' function after evaluating the raw data.
    '''
    if getName(lab) not in os.listdir(os.getcwd()):
        return False, lab, 'No Resource'
    # if lab + "-features.txt" in os.listdir(os.getcwd()):
    #     f = open(lab + "-features.txt", "r")
    #     result = f.readline().split()
    #     f.close()
    #     return True, lab, [float(i) for i in result]

    res = open(getName(lab),"r")
    returnValue = createFile(res, lab)
    res.close()
    return True, lab, returnValue

def readLammpsData(file,col,row):
    f = open(file)
    while True:
        line = f.readline().split()
        if line[0] != "#":
            break
    data = []
    # while line != []:
    for _ in range(row):
        doubles = []
        for j in range(min(col,len(line))):
            doubles.append(float(line[j]))
        data.append(doubles)
        line = f.readline().split()
    return data

def calculateForceStrainCurve(data):
    forceStrainCurve = []
    len0 = data[0][2]

    for dataLine in data:
        vD = dataLine[2]
        fLx = dataLine[3]
        fLy = dataLine[4]
        fLz = dataLine[5]
        fRx = dataLine[6]
        fRy = dataLine[7]
        fRz = dataLine[8]
        lx = dataLine[9]
        ly = dataLine[10]
        lz = dataLine[11]
        axialForce = ((fLx - fRx) * lx + (fLy - fRy) * ly + (fLz - fRz) * lz) / vD
        eps = 100 * (vD - len0) / len0
        forceStrainCurve.append((axialForce,eps))

    return forceStrainCurve



def nrMseLinear(fsCurve, slope=0.5):
    maxY = float("-inf")
    minY = float("inf")
    sumOfSquares = 0

    for yActual, x in fsCurve:
        error = x * slope - yActual
        sumOfSquares += error**2
        maxY = max(yActual, maxY)
        minY = min(yActual, minY)

    rmse = sqrt(sumOfSquares / len(fsCurve))
    return rmse / (maxY - minY)


def calculateReward(file, col, row):
    lammpsData = readLammpsData(file, col, row)
    forceStrainCurve = calculateForceStrainCurve(lammpsData)
    x = nrMseLinear(forceStrainCurve)
    return 1 - x



if __name__ == "__main__":
    rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/candidate"
    for trail in [169,179]:
        print("Calculating Trail %d" % trail)
        trailPath = rootPath + "/" + str(trail)
        for depth in range(1,31):
            depthPath = trailPath + "/Depth" + str(depth)
            for fileName in os.listdir(depthPath):
                if "deformation" in fileName:
                    linkerPath = depthPath + "/" + fileName
                    os.chdir(linkerPath)
                    lab = fileName[:-12]
                    # f = open(lab + "-oldReward.txt", "w")
                    # f.writelines(str(calculateReward(linkerPath + "/" + lab + "-ave-force.d", 13, 200)))
                    # f.close()
                    singleEvaluationTask(lab)


