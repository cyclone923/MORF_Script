import numpy as np


def continous(catagory, trim):
	if trim == False:
		y = np.load("data/BFS/allY/" + catagory + "/labelNoTrim.npy")
		pick_out = np.load("data/BFS/allX/continous_NoTrim.npy")
		print(pick_out[:10])
		positive_idx = np.where(y[:,[1]] == 1)[0]
		negative_idx = np.where(y[:,[0]] == 1)[0]
		print(np.sum(y, axis=0))
		print(positive_idx.shape)
		print(negative_idx.shape)
		np.random.shuffle(positive_idx)
		np.random.shuffle(negative_idx)
		train_idx = np.concatenate((negative_idx[:135000], positive_idx[:15000]))
		test_idx = np.concatenate((negative_idx[135000:], positive_idx[15000:]))
		np.random.shuffle(train_idx)
		np.random.shuffle(test_idx)
		np.save("data/BFS/allY/" + catagory + "/trainIdxNoTrim.npy", pick_out[train_idx])
		np.save("data/BFS/allY/" + catagory + "/testIdxNoTrim.npy", pick_out[test_idx])
	else:
		y = np.load("data/BFS/allY/" + catagory + "/labelTrim.npy")
		pick_out = np.load("data/BFS/allX/continous_Trim.npy")
		print(pick_out[:10])
		positive_idx = np.where(y[:,[1]] == 1)[0]
		negative_idx = np.where(y[:,[0]] == 1)[0]
		print(np.sum(y, axis=0))
		print(positive_idx.shape)
		print(negative_idx.shape)
		np.random.shuffle(positive_idx)
		np.random.shuffle(negative_idx)
		train_idx = np.concatenate((negative_idx[:63000], positive_idx[:7000]))
		test_idx = np.concatenate((negative_idx[63000:], positive_idx[7000:]))
		np.random.shuffle(train_idx)
		np.random.shuffle(test_idx)
		np.save("data/BFS/allY/" + catagory + "/trainIdxTrim.npy", pick_out[train_idx])
		np.save("data/BFS/allY/" + catagory + "/testIdxTrim.npy", pick_out[test_idx])
	

def jump(catagory, trim):
	ct = catagory[-1]
	
	if trim == False:
		y = np.load("data/BFS/allY/" + catagory + "/labelNoTrim.npy")
		pick_out = np.load("data/BFS/allX/jump" + ct + "_Trim.npy")
		print(pick_out[:10])
		positive_idx = np.where(y[:,[1]] == 1)[0]
		negative_idx = np.where(y[:,[0]] == 1)[0]
		print(np.sum(y, axis=0))
		print(positive_idx.shape)
		print(negative_idx.shape)
		np.random.shuffle(positive_idx)
		np.random.shuffle(negative_idx)
		train_idx = np.concatenate((negative_idx[:135000], positive_idx[:15000]))
		test_idx = np.concatenate((negative_idx[135000:], positive_idx[15000:]))
		np.random.shuffle(train_idx)
		np.random.shuffle(test_idx)
		np.save("data/BFS/allY/" + catagory + "/trainIdxNoTrim.npy", train_idx)
		np.save("data/BFS/allY/" + catagory + "/testIdxNoTrim.npy", test_idx)
	else:
		y = np.load("data/BFS/allY/" + catagory + "/labelTrim.npy")
		pick_out = np.load("data/BFS/allX/jump" + ct + "_NoTrim.npy")
		print(pick_out[:10])
		positive_idx = np.where(y[:,[1]] == 1)[0]
		negative_idx = np.where(y[:,[0]] == 1)[0]
		print(np.sum(y, axis=0))
		print(positive_idx.shape)
		print(negative_idx.shape)
		np.random.shuffle(positive_idx)
		np.random.shuffle(negative_idx)
		train_idx = np.concatenate((negative_idx[:63000], positive_idx[:7000]))
		test_idx = np.concatenate((negative_idx[63000:], positive_idx[7000:]))
		np.random.shuffle(train_idx)
		np.random.shuffle(test_idx)
		np.save("data/BFS/allY/" + catagory + "/trainIdxTrim.npy", pick_out[train_idx])
		np.save("data/BFS/allY/" + catagory + "/testIdxTrim.npy", pick_out[test_idx])	

		