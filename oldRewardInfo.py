import os
import numpy as np







rootDir = "/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/candidate"
for numt in range(0,200):
    print("ploting trail %d" % numt)
    trailDir = rootDir + "/" + str(numt)
    avg = []
    std = []
    best = []
    worst = []
    x = []
    for depth in range(1,31):
        depthDir = trailDir + "/Depth" + str(depth)
        rewardInDepth = []
        for fileName in os.listdir(depthDir):
            # print(fileName)

            if "deformation" in fileName:
                labDir = depthDir + "/" + fileName
                # featureFilePath = labDir + '/' + fileName[:-12] + "-features.txt"
                featureFilePath = labDir + '/' + fileName[:-12] + "-oldReward.txt"
                linkerNum = int(fileName[6:-12])
                f = open(featureFilePath,'r')
                line = float(f.readline())
                f.close()
                rewardInDepth.append(line)
        # print(rewardInDepth)


        if rewardInDepth != []:
            avg.append(round(np.mean(rewardInDepth), 2))
            std.append(round(np.std(rewardInDepth), 2))
            best.append(round(np.max(rewardInDepth), 2))
            worst.append(round(np.min(rewardInDepth), 2))
            x.append(depth)
        else:
            print("Depth %d no candidate" % depth)
            avg.append(-1)
            std.append(-1)
            best.append(-1)
            worst.append(-1)
            x.append(depth)



    # print(avg)
    # print(std)
    # print(best)
    # print(worst)
    f = open("/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/" + "oldRewardInfo/" + str(numt) + ".txt", "w")
    f.write("Avg\n")
    for i in range(len(avg)):
        f.write(str(avg[i]))
        if not i == len(avg) - 1:
            f.write(",")
        else:
            f.write("\n")

    f.write("Std\n")
    for i in range(len(std)):
        f.write(str(std[i]))
        if not i == len(std) - 1:
            f.write(",")
        else:
            f.write("\n")

    f.write("Max\n")
    for i in range(len(best)):
        f.write(str(best[i]))
        if not i == len(best) - 1:
            f.write(",")
        else:
            f.write("\n")

    f.write("Min\n")
    for i in range(len(worst)):
        f.write(str(worst[i]))
        if not i == len(worst) - 1:
            f.write(",")
        else:
            f.write("\n")

    f.close()




