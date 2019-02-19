# Python 3
# Pull out information from linker{#}.coeff, linker{#}-{status}.xyz
# Implemented for creating linker{#}-{status}.txt file for molecule command in lammps script serving for fix gcmc
# simulation



import os
import numpy as np

def SetSourceName(lab):
    """obtaining the file name with the information in need

    :param n (int): the ID of the located linker
    :param name (str): the in need status of the linker
    :return: the name for the wanted files
    """
    typeInformation = lab + '.lmpdat'
    return typeInformation


def SetDesName(lab):
    """obtaining the aiming file name

    :param n (int): the ID of the located linker
    :param name (str): the in need status of the linker
    :return: the name for the wanted files
    """
    file = lab + '_generalInformation.txt'
    return file;


def getGeneralInformation(lab, directory):
    sourFile = SetSourceName(lab)
    desFile = SetDesName(lab)
    os.chdir(directory)
    if sourFile not in os.listdir(os.getcwd()):
        return False
    sour = open(sourFile, 'r')
    des = open(desFile,'w+')
    angleFile = open(lab+'angle.txt','r')
    angle = angleFile.readline().split()[0]
    des.write('# ' + desFile + ' molecule file.\n\n')
    generalInformation = sour.readlines()[6:]
    sour.close()

    # Write General Information
    while generalInformation[0] != '\n':
        des.write(generalInformation[0].strip()+'\n')
        generalInformation.pop(0)

    # Read Information from .lmpdat
    while len(generalInformation) != 0:
        if generalInformation[0] == '\n':
            generalInformation.pop(0)

        else:
            if generalInformation[0] == 'Masses\n':
                generalInformation.pop(0)
                mass = []
                while True:
                    try:
                        if generalInformation[0] == '\n':
                            generalInformation.pop(0)
                            continue
                        words = generalInformation[0].split()
                        int(words[0])
                        mass.append(words[0:2])
                        mass[-1][0] = str(mass[-1][0])
                        generalInformation.pop(0)
                    except:
                        break
                continue

            if generalInformation[0] == 'Atoms\n':
                generalInformation.pop(0)
                type = []
                position = []
                while True:
                    try:
                        if generalInformation[0] == '\n':
                            generalInformation.pop(0)
                            continue
                        words = generalInformation[0].split()
                        int(words[0])
                        type.append([words[0],words[2]])
                        position.append([float(words[4]), float(words[5]), float(words[6])])
                        generalInformation.pop(0)
                    except:
                        break
                continue

            generalInformation.pop(0)
            continue
    atomMasses = []
    for e in type:
        for element in mass:
            if e[-1] == element[0]:
                atomMass = float(element[-1])
        atomMasses.append(atomMass)
    length = ((position[0][0]-position[-1][0])**2 + (position[0][1]-position[-1][1])**2 + (position[0][2]-position[-1][2])**2)**0.5

    atomMasses = np.array(atomMasses)
    overAllMass = sum(atomMasses)
    position = np.array(position)
    massCenter = [sum(x) for x in np.matrix.transpose(np.array([i * j for i, j in zip(atomMasses, position)]))] / sum(atomMasses)
    moment = np.array(sum([(x/overAllMass * (np.dot(y - massCenter, y - massCenter) * np.matrix('[1,0,0;0,1,0;0,0,1]') - np.outer(np.matrix.transpose(y-massCenter), (y-massCenter)))) for x,y in zip (atomMasses, position)]))

    des.write('Mass\tlength\tangle\n'+str(overAllMass)+'\t'+str(length)+'\t'+str(angle)+'\nMoment\n')
    for line in moment:
#        print (line)
        for element in line:
            des.write(str(element)+'\t')
        des.write('\n')
    eigenMoment = np.linalg.eigvals(moment)
    des.write('eigenvalue of moment\n')
    for e in eigenMoment:
        des.write(str(e)+'\t')
    des.write('\n')
    des.close()
    return overAllMass, length, angle, moment, eigenMoment

featureList = ["Trail","Depth","lab","Masses", "Length", "Angle"]
os.chdir("/rhome/wzhan097/shared/CleanMORF/randomOutput/Trail200/candidate")
#os.chdir("/Users/weiyi/Downloads/trail")
dir = os.getcwd()
trails = os.listdir(os.getcwd())
processFile = open('generalInfo_process.txt','w+')
for trail in trails:
    try:
        os.chdir(dir+"/"+trail)
        depths = os.listdir(os.getcwd())
        for depth in depths:
            try:
                os.chdir(dir + "/" + trail + "/" + depth)
                linkers = os.listdir(os.getcwd())
                for linker in linkers:
                    words = linker.split(".")
                    if words[-1] == 'lmpdat':
                        lab = words[0]
                        mass, length, angle, moment, eigenMoment = getGeneralInformation(lab, os.getcwd())
                        processFile.write(trail+'\t'+depth+'\t'+lab+'\tdone\n')
                        #print(getGeneralInformation(lab, os.getcwd()))
            except:
                continue
    except:
        continue

