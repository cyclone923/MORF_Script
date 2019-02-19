import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# y = np.load("ML/data/TrailData/hasOutlier/output.npy")
# threshold = np.percentile(np.abs(y),99,0)

colors = ['red' for _ in range(10)]
fname = ['aveCB', 'aveCF', 'aveTB', 'aveTF','ifYesJumpC', 'ifYesJumpT', 'diffOfCT', 'diffOfTC', 'fdNoiseC', 'fdNoiseT']
rootDir = "/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/candidate"
for numt in range(0,20):
    print("ploting trail %d" % numt)
    trailDir = rootDir + "/" + str(numt)
    avg = [[] for _ in range(len(fname))]
    std = [[] for _ in range(len(fname))]
    x = [[] for _ in range(len(fname))]
    for depth in range(1,31):
        depthDir = trailDir + "/Depth" + str(depth)
        rewardInDepth = [[] for _ in range(len(fname))]
        for fileName in os.listdir(depthDir):
            # print(fileName)
            if "deformation" in fileName:
                labDir = depthDir + "/" + fileName
                featureFilePath = labDir + '/' + fileName[:-12] + "-modified-features.txt"
                f = open(featureFilePath,'r')
                line = f.readline().split(",")
                f.close()
                line = map(float,line)
                for i in range(len(fname)):
                    plt.figure(i)
                    # if line[i] <= threshold[i]:
                    plt.scatter(depth, line[i], facecolor='none', edgecolor=colors[i], s=20)
                    rewardInDepth[i].append(line[i])
        for i in range(len(fname)):
            if rewardInDepth[i] != []:
                avg[i].append(np.mean(rewardInDepth[i]))
                std[i].append(np.std(rewardInDepth[i]))
                x[i].append(depth)

    for i in range(len(fname)):
        plt.figure(i)
        plt.title("Trail " + str(numt) + "-" + fname[i])
        plt.plot(x[i],avg[i])
        plt.plot(x[i],std[i])
        plt.legend(["Avg","Std"])
        plt.savefig("Trail200/figure/modifiedFeatures" + "/" + str(numt) + "/" + str(numt) + "-" + fname[i] + ".png")
        plt.close()

