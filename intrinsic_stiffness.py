import numpy as np
import numpy.polynomial.polynomial as poly
import os
import sys

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
    giveUpNo = l - l // nav * nav
    if giveUpNo != 0:
        oriData = oriData[:-(giveUpNo)]
    len0 = oriData[0][2]
    dataReorg = []
    for e in oriData:
        f = [0.5 * (e[3] - e[6]), 0.5 * (e[4] - e[7]), 0.5 * (e[5] - e[8])]
        v = e[15:18]
        mod = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        d = [x / mod for x in v]
        fax = f[0] * d[0] + f[1] * d[1] + f[2] * d[2]
        fn = [f[0] - [x * fax for x in d][0], f[1] - [x * fax for x in d][1], f[2] - [x * fax for x in d][2]]
        mod = (fn[0] ** 2 + fn[1] ** 2 + fn[2] ** 2) ** 0.5
        dataReorg.append([100 * (e[2] - len0) / len0, e[2] - len0, fax])
    data = []
    for i in range(int(len(dataReorg))):
        if i / nav == int(i / nav):
            data.append([])
            for j in range(len(dataReorg[0])):
                data[i // nav].append(0)
                if i // nav != 0:
                    data[i // nav - 1][j] = data[i // nav - 1][j] / nav
        for j in range(len(dataReorg[0])):
            data[i // nav][j] += dataReorg[i][j]

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
    mc = sorted(mc, key=lambda x: x[0])
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

    mc = np.array(mc)
    mt = np.array(mt)

    stiffC = poly.polyfit(mc[:,0],mc[:,2],1)[1]
    stiffT = poly.polyfit(mt[:,0],mt[:,2],1)[1]

    return stiffC, stiffT

def singleTask(lab):
    if lab + "-ave-force.d" not in os.listdir(os.getcwd()) or lab + '-un-squashed.xyz' not in os.listdir(os.getcwd()):
        return False
    res = open(lab + "-ave-force.d","r")
    features = calcFileFeatures(res)
    res.close()
    return features



if __name__ == "__main__":
    # # # d = sys.argv[1]
    # # print("Start")
    # # start = sys.argv[1]
    # # end = sys.argv[2]
    rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/testBFS/finalNode/depth1" #d4 302-6394 d5 6395-171390
    for linkerDir in os.listdir(rootPath):
    # for i in range(int(start),int(end)):
    #     linkerDir = "linker" + str(i)
        os.chdir(rootPath + "/" + linkerDir + "/" + linkerDir + "_deformation")
        stiffC, stiffT= singleTask(linkerDir)
        f = open(linkerDir + '-stiff.txt', 'w')
        f.writelines(str(stiffC)+','+str(stiffT) + "\n")
        f.close()


    # rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/candidate"
    # start = int(sys.argv[1])
    # end = int(sys.argv[2])

    # for t in range(start, end):
    #     for d in range(1, 31):
    #         depthDir = rootPath + "/" + str(t) + "/Depth" + str(d)
    #         for dir in os.listdir(depthDir):
    #             if dir[-11:] == "deformation":
    #                 os.chdir(depthDir + "/" + dir)
    #                 linkerDir = dir[:-12]
    #                 # print(linkerDir)
    #                 stiffC, stiffT = singleTask(linkerDir)
    #                 f = open(linkerDir + '-stiff.txt', 'w')
    #                 f.writelines(str(stiffC)+','+str(stiffT))
    #                 f.close()


    print("End")

