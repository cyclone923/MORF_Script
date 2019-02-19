import sys
import os

des = open("outliers.txt","w+")
featureList = ["Trail","Depth","lab","JumpC", "JumpT", "diffOfCT", "diffOfTC", "aveCB", "aveCF", "aveTB", "aveTF","minRange"]
for e in featureList:
    des.write(e+"\t")
des.write("\n")
os.chdir("/rhome/wzhan097/shared/CleanMORF/randomOutput/Trail200/candidate")
#os.chdir("/Users/weiyi/Downloads/trail")
dir = os.getcwd()
trails = os.listdir(os.getcwd())
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
                        minRange = 0
                        lab = words[0]
                        os.chdir(dir + "/" + trail + "/" + depth + "/" + linker)
                        files = os.listdir(os.getcwd())
                        for file in files:
                            words = file.split("-")
                            if words[-1] == "features.txt":
                                lab = file[:-13]
                                sour = open(file, "r")
                                data = sour.readline().split()
                                des.write(trail + "\t" + depth + "\t" + lab + "\t")
                                for i in range(len(data)):
                                    if abs(float(data[i])) > 100:
                                        des.write(data[i] + "\t")
                                    else:
                                        des.write("---\t")
                            if words[-1] == "range":
                                minRange = 1
                        des.write(str(minRange) + "\t\n")
            except:
                continue
    except:
        continue






