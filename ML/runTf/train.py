from __future__ import division
import network
import tensorflow as tf
import tflearn
import numpy as np
import sys
from util import resetDir
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"



def moment(catagory, trim, order):
	num_features = {2:10, 3:20, 5:56, 10:286}
	n = num_features[order]
	print("Loading Data...")
	
	if trim == False:
		x = np.load("data/BFS/allX/moment.npy")
		y = np.load("data/BFS/allY/" + catagory + "/labelNoTrim.npy")
		pickout_idx = np.load("data/BFS/allX/continous_NoTrim.npy")
		use_idx = np.load("data/BFS/allY/" + catagory + "/trainIdxNoTrim.npy")
	else:
		x = np.load("data/BFS/allX/moment.npy")
		y = np.load("data/BFS/allY/" + catagory + "/labelTrim.npy")
		pickout_idx = np.load("data/BFS/allX/continous_Trim.npy")
		use_idx = np.load("data/BFS/allY/" + catagory + "/trainIdxTrim.npy")
		
	x = x[pickout_idx]

	x = x[:,:,0:n]
	x = x.reshape(x.shape[0], -1)
	x = x[use_idx]
	y = y[use_idx]


	
	k_fold = 5
	positive_idx = np.where(y[:,[1]] == 1)[0]
	negative_idx = np.where(y[:,[0]] == 1)[0]
	fold_size_p = len(positive_idx) // k_fold
	fold_size_n = len(negative_idx) // k_fold


	for k in range(k_fold):
		cut_sp = fold_size_p*k
		cut_ep = fold_size_p*(k+1)
		cut_sn = fold_size_n*k
		cut_en = fold_size_n*(k+1)


		valid_idx = np.concatenate((positive_idx[cut_sp:cut_ep], negative_idx[cut_sn:cut_en]))
		train_idx = np.concatenate((np.delete(positive_idx, [i for i in range(cut_sp, cut_ep)]), 
								   np.delete(negative_idx, [i for i in range(cut_sn, cut_en)])))

		tf.reset_default_graph()
		sys.stdout.flush()
		tflearn.init_graph()

		net = network.moment.create(n, 100)
		
		
		path = catagory + "/moment/" + str(k)
		if trim:
			path += "Trim"
		else:
			path += "NoTrim"
		resetDir("tflearn_logs/" + path)
		resetDir("model/" + path)



		model = tflearn.DNN(net, tensorboard_verbose=0, 
							tensorboard_dir="tflearn_logs/" + path, best_checkpoint_path="model/" + path + "/")
		model.fit(x[train_idx], y[train_idx], n_epoch=20, validation_set=(x[valid_idx], y[valid_idx]), 
				  shuffle=True, show_metric=True, run_id="voxel")
		model.save("model/" + path + "/model.tfl")		
		
		
def point(catagory, trim):
	print("Loading Data...")
	sys.stdout.flush()
	


	if trim == False:
		x = np.load("data/BFS/allX/point.npy")
		y = np.load("data/BFS/allY/" + catagory + "/labelNoTrim.npy")
		pickout_idx = np.load("data/BFS/allX/continous_NoTrim.npy")
		use_idx = np.load("data/BFS/allY/" + catagory + "/trainIdxNoTrim.npy")
	else:
		x = np.load("data/BFS/allX/point.npy")
		y = np.load("data/BFS/allY/" + catagory + "/labelTrim.npy")
		pickout_idx = np.load("data/BFS/allX/continous_Trim.npy")
		use_idx = np.load("data/BFS/allY/" + catagory + "/trainIdxTrim.npy")
		
	x = x[pickout_idx]
	
	
	x = x[use_idx]
	y = y[use_idx]

	
	k_fold = 5
	positive_idx = np.where(y[:,[1]] == 1)[0]
	negative_idx = np.where(y[:,[0]] == 1)[0]
	fold_size_p = len(positive_idx) // k_fold
	fold_size_n = len(negative_idx) // k_fold


	for k in range(k_fold):
		cut_sp = fold_size_p*k
		cut_ep = fold_size_p*(k+1)
		cut_sn = fold_size_n*k
		cut_en = fold_size_n*(k+1)


		valid_idx = np.concatenate((positive_idx[cut_sp:cut_ep], negative_idx[cut_sn:cut_en]))
		train_idx = np.concatenate((np.delete(positive_idx, [i for i in range(cut_sp, cut_ep)]), 
								   np.delete(negative_idx, [i for i in range(cut_sn, cut_en)])))

		
		tf.reset_default_graph()
		sys.stdout.flush()
		tflearn.init_graph()

		net = network.point.create(100)
		
		path = catagory + "/point/" + str(k)
		if trim:
			path += "Trim"
		else:
			path += "NoTrim"
		resetDir("tflearn_logs/" + path)
		resetDir("model/" + path)


		model = tflearn.DNN(net, tensorboard_verbose=0, 
							tensorboard_dir="tflearn_logs/" + path, best_checkpoint_path="model/" + path + "/")
		model.fit(x[train_idx], y[train_idx], n_epoch=50, validation_set=(x[valid_idx], y[valid_idx]), 
				  shuffle=True, show_metric=True, run_id="voxel")
		model.save("model/" + path + "/model.tfl")		
		
def voxel(catagory, trim):
	print("Loading Data...")
	sys.stdout.flush()

	
	if trim == False:
		x = np.load("data/BFS/allX/voxel.npy")
		y = np.load("data/BFS/allY/" + catagory + "/labelNoTrim.npy")
		pickout_idx = np.load("data/BFS/allX/continous_NoTrim.npy")
		use_idx = np.load("data/BFS/allY/" + catagory + "/trainIdxNoTrim.npy")
	else:
		x = np.load("data/BFS/allX/voxel.npy")
		y = np.load("data/BFS/allY/" + catagory + "/labelTrim.npy")
		pickout_idx = np.load("data/BFS/allX/continous_Trim.npy")
		use_idx = np.load("data/BFS/allY/" + catagory + "/trainIdxTrim.npy")
		
	pickout_idx = [i*8+j for j in range(8) for i in pickout_idx]
	x = x[pickout_idx]
	
	y = y[use_idx]
	y = np.repeat(y, 8, axis=0)
	use_idx = [i*8+j for j in range(8) for i in use_idx]
	x = x[use_idx]



	
	k_fold = 5
	positive_idx = np.where(y[:,[1]] == 1)[0]
	negative_idx = np.where(y[:,[0]] == 1)[0]
	fold_size_p = len(positive_idx) // k_fold
	fold_size_n = len(negative_idx) // k_fold
	np.random.shuffle(positive_idx)
	np.random.shuffle(negative_idx)


	for k in range(k_fold):
		cut_sp = fold_size_p*k
		cut_ep = fold_size_p*(k+1)
		cut_sn = fold_size_n*k
		cut_en = fold_size_n*(k+1)


		valid_idx = np.concatenate((positive_idx[cut_sp:cut_ep], negative_idx[cut_sn:cut_en]))
		train_idx = np.concatenate((np.delete(positive_idx, [i for i in range(cut_sp, cut_ep)]), 
								   np.delete(negative_idx, [i for i in range(cut_sn, cut_en)])))


		print("Start Training...")
		sys.stdout.flush()
		
		tf.reset_default_graph()
		sys.stdout.flush()
		tflearn.init_graph()

		net = network.voxel.create(100)
		
		path = catagory + "/voxel/" + str(k)
		if trim:
			path += "Trim"
		else:
			path += "NoTrim"
		resetDir("tflearn_logs/" + path)
		resetDir("model/" + path)


		model = tflearn.DNN(net, tensorboard_verbose=0, 
							tensorboard_dir="tflearn_logs/" + path, best_checkpoint_path="model/" + path + "/")
		model.fit(x[train_idx], y[train_idx], n_epoch=20, validation_set=(x[valid_idx], y[valid_idx]), 
				  shuffle=True, show_metric=True, run_id="voxel")
		model.save("model/" + path + "/model.tfl")		
