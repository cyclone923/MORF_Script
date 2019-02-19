import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




colors = ['blue' for _ in range(10)]
fname = ['ifYesJumpC', 'ifYesJumpT', 'diffOfCT', 'diffOfTC', 'aveCB', 'aveCF', 'aveTB', 'aveTF','fdNoiseC', 'fdNoiseT']

rootDir = "/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/candidate"
for numt in range(200,300):
    print("ploting trail %d" % numt)
    trailDir = rootDir + "/" + str(numt)
    for depth in range(1,31):
        depthDir = trailDir + "/Depth" + str(depth)
        rewardInDepth = []
        for fileName in os.listdir(depthDir):
            # print(fileName)
            if "deformation" in fileName:
                labDir = depthDir + "/" + fileName
                featureFilePath = labDir + '/' + fileName[:-12] + "-modified-features.txt"
                f = open(featureFilePath,'r')
                line = f.readline().split()
                f.close()
                line = map(float,line)
                for i in range(len(line)):
                    plt.figure(i)
                    plt.title("Trail " + str(numt) + "-" + fname[i])
                    plt.scatter(depth, line[i], facecolor='none', edgecolor=colors[i], s=20)
                    rewardInDepth.append(line[i])


    for i in range(len(line)):
        plt.figure(i)
        plt.savefig("Trail200/figure/features" + "/" + str(numt) + "/" + str(numt) + "-" + fname[i] + ".png")
        plt.close()

