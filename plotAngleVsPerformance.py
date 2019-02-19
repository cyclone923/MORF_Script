import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable






rootDir = "/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/candidate"


cm = matplotlib.cm.get_cmap('RdYlBu')

for numt in range(0,200):
    print("ploting trail %d" % numt)
    trailDir = rootDir + "/" + str(numt)
    fig = plt.figure()
    plt.title("Carboxy-Angle Vs Performance")

    x = []
    y = []
    colors = []

    for depth in range(1,31):
        depthDir = trailDir + "/Depth" + str(depth)
        # print(depth)
        for fileName in os.listdir(depthDir):
            # print(fileName)
            if "deformation" in fileName:
                labDir = depthDir + "/" + fileName
                # featureFilePath = labDir + '/' + fileName[:-12] + "-features.txt"
                featureFilePath = labDir + '/' + fileName[:-12] + "-oldReward.txt"
                linkerNum = int(fileName[6:-12])
                f = open(featureFilePath,'r')
                line = f.readline()
                f.close()
                performance = float(line)
                angleFilePath = depthDir + '/' + fileName[:-12] + "angle.txt"
                linkerNum = int(fileName[6:-12])
                f = open(angleFilePath,'r')
                line = f.readline()
                angel = float(line)
                f.close()
                x.append(angel)
                y.append(performance)
                colors.append(depth)
                # print("Angle: %f, Performance: %f, LinkerNum: %d" % (angel, performance, linkerNum))

    sc = plt.scatter(x, y, c=colors, cmap=cm, s=20)
    cbar = plt.colorbar(sc)
    cbar.set_label("Depth")
    fig.savefig("Trail200/figure/features" + "/" + str(numt) + "/" + str(numt) + "-" + "AngleVsPerformance" + ".png")
    plt.close()


