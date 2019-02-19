# Python 3
# Pull out information from linker{#}-ave-force.d, Calculate the needed results, fit the data to a function and shift
# the curve through zero point and return the features of the curves



plotting = 'no'  #if = 'yes', will get the plots
writingFile = 'no' #if = 'yes', will get the files saved
errorFile = 'yes' #if = 'yes', will get the error files saved

import math
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import os
import sys
import scipy.interpolate as inter
import pylab

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
    minS = 200
    maxS = 5000
    while maxS-minS > 5:
        delta = (maxS-minS)/32.0
        minS, maxS = findMatchRange(x, y, delta, minS, maxS)
        if minS == 'False':
            return minS, maxS

    while minS < 1000 and threshold > 20:
        threshold = threshold / 1.2
        minS = 200
        maxS = 2000
        while maxS - minS > 5:
            delta = (maxS - minS) / 32.0
            minS, maxS = findMatchRange(x, y, delta, minS, maxS)
    while minS < 500 and threshold > 10:
        threshold = threshold / 1.1
        minS = 200
        maxS = 1000
        while maxS - minS > 5:
            delta = (maxS - minS) / 32.0
            minS, maxS = findMatchRange(x, y, delta, minS, maxS)
    return minS, maxS

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
    coeffs = poly.polyfit(mcn[:,0],mcn[:,-2],5)
    pfit = poly.Polynomial(coeffs)

    m = np.array(list(map(lambda x,y:[x,y],mcn[:,0],mcn[:,-2])))
    if errorFile == 'yes' or errorFile == 'Yes':
        if lab + "-errors.txt" in os.listdir(os.getcwd()):
            os.system("rm " + os.getcwd() + '/' + lab + "-errors.txt")
    # Shift the curve to satisfy the rule of lowest energy as 0 and at 0% strain
    #eo, PEo = findMin(m, pfit)
    #mcn[:,0] -= eo
    #mtn[:,0] -= eo
    #mcn[:,-2] -= PEo
    #mtn[:,-2] -= PEo
    m = np.array(list(map(lambda x, y: [x, y], mcn[:, 0], mcn[:, -2])))
    # Plot FD and PE curve after shifted and save the shifted data points
    '''
    if writingFile == 'yes' or writingFile == 'Yes':
        des = open(lab + "-FD-compression-polyfit.d", "w+")
        for e in mcn:
            for i in range(len(e)):
                if i != len(e) - 1:
                    des.write(str(e[i]) + "\t")
                else:
                    des.write(str(e[i]) + "\n")
        des.close()
        des = open(lab + "-FD-tension-polyfit.d", "w+")
        for e in mtn:
            for i in range(len(e)):
                if i != len(e) - 1:
                    des.write(str(e[i]) + "\t")
                else:
                    des.write(str(e[i]) + "\n")
        des.close()
    '''
    if plotting == 'yes' or plotting == 'Yes':
        fig = plt.figure(figsize=(8.5, 12))
        ax1 = fig.add_subplot(311)
        ax1.plot(mcn[:, 0], mcn[:, 2], label='Normal Force - Compression', linewidth=0.5)
        ax1.plot(mcn[:, 0], mcn[:, 1], label='Axial Force - Compression', linewidth=0.5)
        ax1.plot(mtn[:, 0], mtn[:, 2], label='Normal Force - Tension', linewidth=0.5)
        ax1.plot(mtn[:, 0], mtn[:, 1], label='Axial Force - Tension', linewidth=0.5)
        ax1.legend()
        ax1.set_xlabel('Strain (%)', family='DejaVu Sans', fontsize=10, color='black')
        ax1.set_ylabel('Force (Kcal/mol/A)', family='DejaVu Sans', fontsize=10, color='black')
        ax21 = fig.add_subplot(312)
        ax21.plot(mcn[:, 0], mcn[:, -3], label='Potential Energy - Compression', linewidth=0.5)
        ax21.plot(mtn[:, 0], mtn[:, -3], label='Potential Energy - Tension', linewidth=0.5)
        ax21.legend(loc=1)
        ax21.set_ylabel('Potential Energy (Kcal/mol)', family='DejaVu Sans', fontsize=10, color='black')
        ax22 = ax21.twinx()
        ax22.plot(mcn[:, 0], mcn[:, -2], 'r', label='Total Energy - Compression', linewidth=0.5)
        ax22.plot(mtn[:, 0], mtn[:, -2], 'g', label='Total Energy - Tension', linewidth=0.5)
        ax22.legend(loc=2)
        ax22.set_ylabel('Total Energy (Kcal/mol/A)', family='DejaVu Sans', fontsize=10, color='black')
        ax22.set_title('')
        ax31 = fig.add_subplot(313)
        ax31.plot(mcn[:, 0], mcn[:, -1], label='Temp - Compression', linewidth=0.5)
        ax31.plot(mtn[:, 0], mtn[:, -1], label='Temp - Tension', linewidth=0.5)
        ax31.legend(loc=1)
        ax31.set_ylabel('Potential Energy (Kcal/mol)', family='DejaVu Sans', fontsize=10, color='black')
        ax32 = ax31.twinx()
        ax32.plot(mcn[:, 0], mcn[:, -2], 'r', label='Total Energy - Compression', linewidth=0.5)
        ax32.plot(mtn[:, 0], mtn[:, -2], 'g', label='Total Energy - Tension', linewidth=0.5)
        ax32.legend(loc=2)
        ax32.set_ylabel('Total Energy (Kcal/mol/A)', family='DejaVu Sans', fontsize=10, color='black')
        ax32.set_title('')
        plt.savefig(lab + "-FD-TE-curve.jpeg")


    # Get the spline function of FD curve for further evaluation and plot the fitted curve

    minSC, maxS = findBestMatch(mcn[:, 0], mcn[:, 2])
    fdFunctionC = inter.UnivariateSpline(mcn[:, 0], mcn[:, 2], s = maxS)
    if errorFile == 'yes' or errorFile == 'Yes':
        if minSC == 'False':
            des = open(os.getcwd() + '/' + lab + "-errors.txt",'a+')
            des.write("Compression-FD-curve-too-noisy\n")
            des.close()
    fdNoiseC = maxS
    minST, maxS = findBestMatch(mtn[:, 0], mtn[:, 2])
    fdFunctionT = inter.UnivariateSpline(mtn[:, 0], mtn[:, 2], s = maxS)
    if errorFile == 'yes' or errorFile == 'Yes':
        if minST == 'False':
            des = open(os.getcwd() + '/' + lab + "-errors.txt",'a+')
            des.write("Tension-FD-curve-too-noisy\n")
            des.close()
    fdNoiseT = maxS

    if plotting =='yes' or plotting =='Yes':
        plt.figure()
        plt.plot(mcn[:, 0], fdFunctionC(mcn[:, 0]), label='Normal Force - Compression', linewidth=0.5)
        plt.plot(mtn[:, 0], fdFunctionT(mtn[:, 0]), label='Normal Force - Tension', linewidth=0.5)
        plt.savefig(lab + "-fitting.jpeg")

    # Calculate the flexibility of the molecule with 'flexibility = PresentLength*Delta(Force)/Delta(Length)'
    # plot the flexibility information and save the data
    flexibilityC = []
    flexibilityT = []
    if writingFile =='yes' or writingFile =='Yes':
        des = open(lab + "-FD-Stiffness.d","w+")
        des.write("strain\tcompression\ttension\n")
    _from = min(m[:,0])
    _to = max(m[:,0])
    step = (_to - _from)/1000
    xPoints = [x*step + _from for x in range(1000+1)]
    deltaX = 0.0001

    # Get the differences of tension and compression process and evaluate the jump abnormal situation
    _diffOfCT_ = 0
    _diffOfTC_ = 0
    _jumpC_=[]
    _jumpT_=[]
    jumpDepthThreshold=5
    jumpWidthThreshold=1
    jumpStartC = float('-inf')
    jumpStartT = float('-inf')
    jumpEndC = float('-inf')
    jumpEndT = float('-inf')
    for e in xPoints:
        flexibilityC.append([e,(fdFunctionC(e+deltaX)-fdFunctionC(e))/deltaX * ((100+e)/100.0)])
        flexibilityT.append([e,(fdFunctionT(e+deltaX)-fdFunctionT(e))/deltaX * ((100+e)/100.0)])
        #flexibilityC.append([e,(abs(fdFunctionC(e+deltaX)-fdFunctionC(e)))/deltaX * ((100+e)/100.0)])
        #flexibilityT.append([e,(abs(fdFunctionT(e+deltaX)-fdFunctionT(e)))/deltaX * ((100+e)/100.0)])

        if flexibilityC[-1][1] < -1:   #The Threshold was set randomly, in need of more trials.
            if jumpStartC==float('-inf'):
                jumpStartC = e
            else:
                jumpEndC = e
        else:
            if jumpStartC!=float('-inf') and jumpEndC!=float('-inf'):
                if (fdFunctionC(jumpStartC)-fdFunctionC(jumpStartC+jumpWidthThreshold))>=2 and (fdFunctionC(jumpStartC)-fdFunctionC(jumpEndC))>jumpDepthThreshold:
                    _jumpC_.append([jumpStartC,jumpEndC-jumpStartC,fdFunctionC(jumpStartC)-fdFunctionC(jumpEndC)])
            jumpStartC = float('-inf')
            jumpEndC = float('-inf')
        if flexibilityT[-1][1] < -1:   #The Threshold was set randomly, in need of more trials.
            if jumpStartT==float('-inf'):
                jumpStartT = e
            else:
                jumpEndT = e
        else:
            if jumpStartT!=float('-inf') and jumpEndT!=float('-inf'):
                if (fdFunctionT(jumpStartT)-fdFunctionT(jumpStartT+jumpWidthThreshold))>=2 and (fdFunctionT(jumpStartT)-fdFunctionT(jumpEndT))>jumpDepthThreshold:
                    _jumpT_.append([jumpStartT,jumpEndT-jumpStartT,fdFunctionT(jumpStartT)-fdFunctionT(jumpEndT)])
            jumpStartT = float('-inf')
            jumpEndT = float('-inf')

        if fdFunctionC(e) - fdFunctionT(e) > _diffOfCT_:
            _diffOfCT_ = fdFunctionC(e) - fdFunctionT(e)
        if fdFunctionT(e) - fdFunctionC(e) > _diffOfTC_:
            _diffOfTC_ = fdFunctionT(e) - fdFunctionC(e)

        if writingFile == 'yes' or writingFile == 'Yes':
            des.write(str(e) + '\t' + str(flexibilityC[-1][1]) + '\t' + str(flexibilityT[-1][1]) + '\n')

    i = 0
    while i < len(_jumpC_)-1:
        if _jumpC_[i+1][0] - (_jumpC_[i][0]+_jumpC_[i][1]) < 2:
            if fdFunctionC(_jumpC_[i][0]+_jumpC_[i][0])-fdFunctionC(_jumpC_[i+1][1]+_jumpC_[i+1][0]) > 0:
                _jumpC_[i+1]=[_jumpC_[i][0],_jumpC_[i+1][1],fdFunctionC(_jumpC_[i][0])-fdFunctionC(_jumpC_[i+1][1])]
                _jumpC_.pop(i)
            else:
                i +=1
        else:
            i += 1

    i = 0
    while i < len(_jumpT_) - 1:
        if _jumpT_[i + 1][0] - (_jumpT_[i][0] + _jumpT_[i][1]) < 2:
            if fdFunctionT(_jumpT_[i][0]+_jumpT_[i][0]) - fdFunctionT(_jumpT_[i + 1][1]+_jumpT_[i+1][0]) > 0:
                _jumpT_[i + 1] = [_jumpT_[i][0], _jumpT_[i + 1][1],fdFunctionT(_jumpT_[i][0]) - fdFunctionT(_jumpT_[i + 1][1])]
                _jumpT_.pop(i)
            else:
                i +=1
        else:
            i += 1


    # Calculate the average of flexibility before and after bulk


    _countCF_ = 1
    _countTF_ = 1
    _countCB_ = 1
    _countTB_ = 1
    aveThresholdC = 40
    aveThresholdT = 40
    _aveCF_ = flexibilityC[aveThresholdC][1]
    _aveTF_ = flexibilityT[aveThresholdT][1]
    _aveCB_ = flexibilityC[-(1+aveThresholdC)][1]
    _aveTB_ = flexibilityT[-(1+aveThresholdT)][1]
    rangeCF=[aveThresholdC]
    rangeTF=[aveThresholdT]
    rangeCB=[-(1+aveThresholdC)]
    rangeTB=[-(1+aveThresholdT)]
    aveThreshold = max([aveThresholdC,aveThresholdT])
    for i in range(aveThreshold,len(flexibilityC) - 1):
        if i<200 and (_countCF_ < aveThresholdC or abs(flexibilityC[i + 1][1] - (_aveCF_/_countCF_)) < 1000/fdNoiseC):    #The Threshold was set randomly, in need of more trials.
            _aveCF_ += flexibilityC[i + 1][1]
            rangeCF.append(i + 1)
            _countCF_ += 1
        if i<200 and (_countCB_ < aveThresholdC*3 or abs(flexibilityC[-i - 2][1] - (_aveCB_/_countCB_)) < 1000/fdNoiseC):    #The Threshold was set randomly, in need of more trials.
            _aveCB_ += flexibilityC[-i - 2][1]
            rangeCB.append(-i - 2)
            _countCB_ += 1
        if i<200 and (_countTF_ < aveThresholdT or abs(flexibilityT[i + 1][1] - (_aveTF_/_countTF_)) < 1000/fdNoiseT):    #The Threshold was set randomly, in need of more trials.
            _aveTF_ += flexibilityT[i + 1][1]
            rangeTF.append(i + 1)
            _countTF_ += 1
        if i<200 and (_countTB_ < aveThresholdT*3 or abs(flexibilityT[-i - 2][1] - (_aveTB_/_countTB_)) < 1000/fdNoiseT):    #The Threshold was set randomly, in need of more trials.
            _aveTB_ += flexibilityT[-i - 2][1]
            rangeTB.append(-i - 2)
            _countTB_ += 1
    _aveCB_ = _aveCB_ / _countCB_
    _aveCF_ = _aveCF_ / _countCF_
    _aveTB_ = _aveTB_ / _countTB_
    _aveTF_ = _aveTF_ / _countTF_

    if abs(_aveCB_- flexibilityC[-(1+aveThresholdC)][1]) > 1000/fdNoiseC:    #The Threshold was set randomly, in need of more trials.
        i=0
        while abs(_aveCB_- flexibilityC[-(aveThresholdC+1+i)][1]) > 1000/fdNoiseC and i < _countCB_ - 20:
            _aveCB_ = (_aveCB_ * (_countCB_ - i) - flexibilityC[-(aveThresholdC+1+i)][1])/(_countCB_-1-i)
            i = i+1
            rangeCB.pop(0)
        if errorFile == 'yes' or errorFile == 'Yes':
            des = open(os.getcwd() + '/' + lab + "-errors.txt",'a+')
            des.write("Compression-FD-curve-High-Stiffness-Inaccurate\n")
            des.close()
    if abs(_aveCF_- flexibilityC[aveThresholdC][1]) > 1000/fdNoiseC:    #The Threshold was set randomly, in need of more trials.
        i=0
        while abs(_aveCF_- flexibilityC[aveThresholdC+i][1]) > 1000/fdNoiseC and i < _countCF_ - 20:
            _aveCF_ = (_aveCF_ * (_countCF_ - i) - flexibilityC[aveThresholdC+i][1])/(_countCF_-1-i)
            i = i + 1
            rangeCF.pop(0)
        if errorFile == 'yes' or errorFile == 'Yes':
            des = open(os.getcwd() + '/' + lab + "-errors.txt",'a+')
            des.write("Compression-FD-curve-Low-Stiffness-Inaccurate\n")
            des.close()
    if abs(_aveTB_- flexibilityT[-(1+aveThresholdT)][1]) > 1000/fdNoiseT:    #The Threshold was set randomly, in need of more trials.
        i=0
        while abs(_aveTB_- flexibilityT[-(1+aveThresholdT+i)][1]) > 1000/fdNoiseT and i < _countTB_ - 20:
            _aveTB_ = (_aveTB_ * (_countTB_ - i) - flexibilityT[-(1+aveThresholdT+i)][1])/(_countTB_-1-i)
            i = i + 1
            rangeTB.pop(0)
        if errorFile == 'yes' or errorFile == 'Yes':
            des = open(os.getcwd() + '/' + lab + "-errors.txt",'a+')
            des.write("Tension-FD-curve-High-Stiffness-Inaccurate\n")
            des.close()
    if abs(_aveTF_- flexibilityT[aveThresholdT][1]) > 1000/fdNoiseT:    #The Threshold was set randomly, in need of more trials.
        i = 0
        while abs(_aveTF_ - flexibilityT[aveThresholdT + i][1]) >1000/fdNoiseT and i < _countTF_ - 20:
            _aveTF_ = (_aveTF_ * (_countTF_ - i) - flexibilityT[aveThresholdT + i][1]) / (_countTF_ - 1 - i)
            i = i + 1
            rangeTF.pop(0)
        if errorFile == 'yes' or errorFile == 'Yes':
            des = open(os.getcwd() + '/' + lab + "-errors.txt",'a+')
            des.write("Tension-FD-curve-Low-Stiffness-Inaccurate\n")
            des.close()

    flexibilityT = np.array(flexibilityT)
    flexibilityC = np.array(flexibilityC)

    if plotting =='yes' or plotting =='Yes':
        plt.figure(figsize=(8.5, 5))
        plt.plot(flexibilityC[:, 0], flexibilityC[:, 1], label='Flexibility - Compression', linewidth=0.5)
        plt.plot(flexibilityT[:, 0], flexibilityT[:, 1], label='Flexibility - Tension', linewidth=0.5)
        plt.legend()
        plt.xlabel('Strain (%)', family='DejaVu Sans', fontsize=10, color='black')
        plt.ylabel('Flexibility (Kcal/mol/A)', family='DejaVu Sans', fontsize=10, color='black')
        plt.xticks(family='DejaVu Sans', fontsize=10, color='black')
        plt.yticks(family='DejaVu Sans', fontsize=10, color='black')
        plt.savefig(lab + "-Stiffness-curve.jpeg")

    errorC = 0
    errorT = 0
    for i in range(len(mcn)):
        errorC += abs(mcn[i][2]-fdFunctionC(mcn[i][0]))
    errorC = errorC/len(mcn)
    for i in range(len(mtn)):
        errorT += abs(mtn[i][2]-fdFunctionT(mtn[i][0]))
    errorT = errorT/len(mtn)

    cx1 = 0
    cx2 = 0
    tx1 = 0
    tx2 = 0
    cy1 = 0
    cy2 = 0
    ty1 = 0
    ty2 = 0
    for e in rangeCF:
        cx1 += xPoints[e]
        cy1 += fdFunctionC(xPoints[e])
    cx1 = cx1/len(rangeCF)
    cy1 = cy1/len(rangeCF)
    for e in rangeTF:
        tx1 += xPoints[e]
        ty1 += fdFunctionT(xPoints[e])
    tx1 = tx1/len(rangeTF)
    ty1 = ty1/len(rangeTF)
    for e in rangeCB:
        cx2 += xPoints[e]
        cy2 += fdFunctionC(xPoints[e])
    cx2 = cx2/len(rangeCB)
    cy2 = cy2/len(rangeCB)
    for e in rangeTB:
        tx2 += xPoints[e]
        ty2 += fdFunctionT(xPoints[e])
    tx2 = tx2/len(rangeTB)
    ty2 = ty2/len(rangeTB)
    intersectionC = (cy1-cy2+_aveCB_*cx2-_aveCF_*cx1)/(_aveCB_-_aveCF_)
    intersectionT = (ty1-ty2+_aveTB_*tx2-_aveTF_*tx1)/(_aveTB_-_aveTF_)



    return _jumpC_, _jumpT_, _diffOfCT_, _diffOfTC_, _aveCB_, _aveCF_, _aveTB_, _aveTF_,intersectionC,intersectionT, errorC*10/(fdNoiseC**0.5), errorT*10/(fdNoiseT**0.5)

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
        v = e[15:18]
        mod = (v[0]**2+v[1]**2+v[2]**2)**0.5
        d = [x / mod for x in v]
        fax = f[0]*d[0]+f[1]*d[1]+f[2]*d[2]
        fn = [f[0]-[x * fax for x in d][0],f[1]-[x * fax for x in d][1],f[2]-[x * fax for x in d][2]]
        mod = (fn[0]**2+fn[1]**2+fn[2]**2)**0.5
        dataReorg.append([100*(e[2]-len0)/len0,e[2]-len0,fax,mod,e[18],e[19],e[20]])
    data = []
    for i in range(int(len(dataReorg))):
        if i/nav == int(i/nav):
            data.append([])
            for j in range(len(dataReorg[0])):
                data[i//nav].append(0)
                if i//nav != 0:
                    data[i // nav - 1][j] = data[i // nav - 1][j]/nav
        for j in range(len(dataReorg[0])):
            data[i//nav][j]+=dataReorg[i][j]

    for j in range(len(dataReorg[0])):
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

    return createFits(mc, mt, lab)

def getName(lab):
    filename = lab + "-ave-force.d"
    return filename

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
    if lab + "-FD-PE-curve.jpeg" in os.listdir(os.getcwd()):
        res = open(getName(lab), "r")
        returnValue = createFile(res, lab)
        res.close()
        return 'Redo', lab, returnValue
    res = open(getName(lab),"r")
    returnValue = createFile(res, lab)
    res.close()
    return True, lab, returnValue

if __name__ == "__main__":
    # d = sys.argv[1]
    # start = sys.argv[1]
    # end = sys.argv[2]
    # rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/testBFS/finalNode/depth2"  #d4 302-6394 d5 6395-171390
    # for linkerDir in os.listdir(rootPath):
    # # for i in range(int(start),int(end)):
    # #     linkerDir = "linker" + str(i)
    #     os.chdir(rootPath + "/" + linkerDir + "/" + linkerDir + "_deformation")
    #     lab = linkerDir
    #     result = singleEvaluationTask(lab)
    #     # print(result)
    #     # Suggest saving format:
    #     if result[0]!= False:
    #         # errorC*10/(fdNoiseC**0.5), errorT*10/(fdNoiseT**0.5)
    #         data = result[2]
    #         des = open(lab+'-features.txt','w+')
    #         des.write('Jumps in compression dir:\nNumber of jumps:'+str(len(data[0]))+'\n')
    #         for e in data[0]:
    #             des.write(str(e[0])+'\t'+str(e[1])+'\t'+str(e[2])+'\n')
    #         des.write('Jumps in Tension dir:\nNumber of jumps:'+str(len(data[1]))+'\n')
    #         for e in data[1]:
    #             des.write(str(e[0])+'\t'+str(e[1])+'\t'+str(e[2])+'\n')

    #         des.write('Max C>T forces difference:\n'+str(data[2])+'\n')
    #         des.write('Max T>C forces difference:\n'+str(data[3])+'\n')
    #         des.write('Stiffness of stretched molecule in compression dir:\n'+str(data[4])+'\n')
    #         des.write('Stiffness of squashed molecule in compression dir:\n'+str(data[5])+'\n')
    #         des.write('Stiffness of stretched molecule in tension dir:\n'+str(data[6])+'\n')
    #         des.write('Stiffness of squashed molecule in tension dir:\n'+str(data[7])+'\n')
    #         des.write('Strain of compression curve intersection:\n'+str(data[8])+'\n')
    #         des.write('Strain of tension curve intersection:\n'+str(data[9])+'\n')
    #         des.write('A value describes the noisy level of compression dir:\n'+str(data[10])+'\n')
    #         des.write('A value describes the noisy level of tension dir:\n'+str(data[11])+'\n')


    
    rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/candidate"
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    for t in range(start, end):
        for d in range(1, 31):
            depthDir = rootPath + "/" + str(t) + "/Depth" + str(d)
            for dir in os.listdir(depthDir):
                if dir[-11:] == "deformation":
                    os.chdir(depthDir + "/" + dir)
                    linkerDir = dir[:-12]
                    lab = linkerDir
                    result = singleEvaluationTask(lab)
                    # print(result)
                    # Suggest saving format:
                    if result[0]!= False:
                        # errorC*10/(fdNoiseC**0.5), errorT*10/(fdNoiseT**0.5)
                        data = result[2]
                        des = open(lab+'-features.txt','w+')
                        des.write('Jumps in compression dir:\nNumber of jumps:'+str(len(data[0]))+'\n')
                        for e in data[0]:
                            des.write(str(e[0])+'\t'+str(e[1])+'\t'+str(e[2])+'\n')
                        des.write('Jumps in Tension dir:\nNumber of jumps:'+str(len(data[1]))+'\n')
                        for e in data[1]:
                            des.write(str(e[0])+'\t'+str(e[1])+'\t'+str(e[2])+'\n')

                        des.write('Max C>T forces difference:\n'+str(data[2])+'\n')
                        des.write('Max T>C forces difference:\n'+str(data[3])+'\n')
                        des.write('Stiffness of stretched molecule in compression dir:\n'+str(data[4])+'\n')
                        des.write('Stiffness of squashed molecule in compression dir:\n'+str(data[5])+'\n')
                        des.write('Stiffness of stretched molecule in tension dir:\n'+str(data[6])+'\n')
                        des.write('Stiffness of squashed molecule in tension dir:\n'+str(data[7])+'\n')
                        des.write('Strain of compression curve intersection:\n'+str(data[8])+'\n')
                        des.write('Strain of tension curve intersection:\n'+str(data[9])+'\n')
                        des.write('A value describes the noisy level of compression dir:\n'+str(data[10])+'\n')
                        des.write('A value describes the noisy level of tension dir:\n'+str(data[11])+'\n')
                    
    print("End")