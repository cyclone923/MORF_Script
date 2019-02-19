import os
import shutil


def resetDir(path):
	if os.path.exists(path):
		shutil.rmtree(path)
		os.makedirs(path)
	else:
		os.makedirs(path)