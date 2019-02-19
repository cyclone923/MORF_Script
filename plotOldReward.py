import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np







colors = ['y']
fname = ['oldReward']

rootDir = "/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/candidate"
for numt in range(0,200):
    print("ploting trail %d" % numt)
    trailDir = rootDir + "/" + str(numt)
    avg = []
    std = []
    x = []
    best = []
    for depth in range(1,31):
        depthDir = trailDir + "/Depth" + str(depth)
        rewardInDepth = []
        angel = []
        for fileName in os.listdir(depthDir):
            # print(fileName)

            if "deformation" in fileName:
                labDir = depthDir + "/" + fileName
                # featureFilePath = labDir + '/' + fileName[:-12] + "-features.txt"
                featureFilePath = labDir + '/' + fileName[:-12] + "-oldReward.txt"
                linkerNum = int(fileName[6:-12])
                f = open(featureFilePath,'r')
                line = f.readline().split()
                f.close()
                line = [float(format(float(i), '.2f')) for i in line]
                for i in range(len(line)):
                    plt.figure(i)
                    plt.title("Trail " + str(numt) + "-" + fname[i])
                    plt.scatter(depth, line[i], facecolor='none', edgecolor=colors[i], s=20)
                    rewardInDepth.append((line[i], linkerNum))
            elif "angle" in fileName:
                angleFilePath = depthDir + '/' + fileName[:-9] + "angle.txt"
                linkerNum = int(fileName[6:-9])
                f = open(angleFilePath,'r')
                line = f.readline()
                angel.append((float(line),linkerNum))
                f.close()
        # print(rewardInDepth)
        # print(angel)


        if rewardInDepth != []:
            avg.append(np.mean([i[0] for i in rewardInDepth]))
            std.append(np.std([i[0] for i in rewardInDepth]))
            x.append(depth)
            bestLinkerNum = max(angel)[1]
            if len(angel) != len(rewardInDepth):
                print(angel)
                print(rewardInDepth)
                exit(0)
            for i in rewardInDepth:
                if i[1] == bestLinkerNum:
                    best.append(i[0])

    plt.figure(0)
    plt.plot(x,avg)
    plt.plot(x,std)
    # print(x)
    # print(best)
    # exit(0)
    plt.plot(x,best)
    plt.legend(["Avg", "Std", "Most Linear Linker"])
    for i in range(len(fname)):
        plt.figure(i)
        plt.savefig("Trail200/figure/features" + "/" + str(numt) + "/" + str(numt) + "-" + fname[i] + ".png")
        plt.close()


# plt.legend(label)
# plt.show()