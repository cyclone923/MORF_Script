import numpy as np
import os
import sys

	
def getMoment(dir, linkerNum):
	return np.load(dir + "/linker" + str(linkerNum) + "-classified-moments.npy")
	

def getPoint(dir, linkerNum):
	cloud = np.load(dir + "pointCloud.npy")
	dups = [1 for _ in range(cloud.shape[0])]
	dups[-1] += 57-cloud.shape[0]
	repeat = np.repeat(cloud, dups, axis=0)
	return repeat
	
def getVoxel(dir, linkerNum):
	dir = dir + "augmentVoxel7"
	all_voxel = np.zeros(shape=(8,4,17,17,15))
	for i in range(8):
		voxelDir = dir + "/" + str(i) + ".npy"
		v = np.load(voxelDir)
		all_voxel[i,:,:v.shape[1],:v.shape[2],:v.shape[3]] = v
	return all_voxel
		
def getStiff(dir, linkerNum):
    f = open(dir + "/linker" + str(linkerNum) + "-stiff.txt")
    line = f.readline().split(",")
    f.close()
    return np.array(list(map(float,line)))	

def getFeatures(dir, linkerNum):
    f = open(dir + "/linker" + str(linkerNum) + "-features.txt")
    features = []
    while True:
        line = f.readline().split()
        if line == []:
            break
        if line[0] == "Number":
            features.append(line[-1].split(":")[1])
        if len(line) == 1:
            features.append(line[0])
    return np.array(list(map(float,features)))

def getAngle(dir, linkerNum):
	return np.load(dir + "carboxAngle.npy")
	
def getBBRatio(dir, linkerNum):
	return np.load(dir + "bbRatio.npy")
	

	
	
moment = np.zeros(shape=(171390,4,286), dtype=np.float16)	
point = np.zeros(shape=(171390,57,8), dtype=np.float16)
voxel = np.zeros(shape=(171390*8,4,17,17,15), dtype=np.float16)
stiff = np.zeros(shape=(171390,2), dtype=np.float16)
features = np.zeros(shape=(171390,12), dtype=np.float16)
angle = np.zeros(shape=(171390), dtype=np.float16)
bbRatio = np.zeros(shape=(171390), dtype=np.float16)


rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/BFS/finalNode"

data_dict = {"moment": moment, "point": point, "voxel": voxel, "stiff":stiff,
			 "features": features, "angle": angle, "bbRatio":bbRatio}
			 
task_dict = {"moment": True, "point": False, "voxel": False, "stiff": False,
			 "features": False, "angle": False, "bbRatio": False}
			 
fuction_dict = {"moment": getMoment, "point": getPoint, "voxel": getVoxel, "stiff": getStiff,
			 "features": getFeatures, "angle": getAngle, "bbRatio": getBBRatio}

i = 0
for d in range(1,6):
	depthDir = rootPath + "/depth" + str(d)
	for dir in os.listdir(depthDir):
		sys.stdout.flush()
		linkerNum = int(dir[6:])
		if linkerNum < 171390:
			dir = depthDir + "/" + dir + "/" + dir + "_deformation/"
			for t in task_dict:
				if task_dict[t] == True:
					array = fuction_dict[t](dir, linkerNum)
					if t != "voxel":
						data_dict[t][linkerNum] = array
					else:
						data_dict[t][linkerNum*8:linkerNum*8+8] = array
		i += 1
		print(i)

	
for name in task_dict:
	if task_dict[name] == True:
		np.save("BFS/" + name + ".npy", data_dict[name])

						
						
			
		

