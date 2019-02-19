import os

smiDir = "/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/outputSMILE"
candDir = "/rhome/yangchen/shared/CleanMORF/randomOutput/Trail200/candidate"
for i in range(200,500):
    f = open(smiDir + "/" + str(i) + ".txt")
    while True:
        line = f.readline().split()
        if line == []:
            break
        if line[0] == "Iteration:":
            t = line[1]
        if line[0] == "Depth":
            d = line[1]
        if line[0] == "SMILE:":
            lab = line[3]
            smi = line[1]
            fw = open(candDir + "/" + t + "/Depth" + d + "/" + lab + "_deformation/" + lab + "-smi.txt", 'w')
            fw.writelines(smi)
            fw.close()
    f.close()
