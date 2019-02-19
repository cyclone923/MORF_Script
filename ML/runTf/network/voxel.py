import tflearn



def create(bs):
	net = tflearn.input_data(shape=[None, 17, 17, 15, 4])
	net = tflearn.conv_3d(net, nb_filter=8, filter_size=5, activation="relu", strides=1, padding="valid")
	net = tflearn.conv_3d(net, nb_filter=16, filter_size=5, activation="relu", strides=1, padding="valid")
	net = tflearn.conv_3d(net, nb_filter=32, filter_size=3, activation="relu", strides=1, padding="valid")
	net = tflearn.conv_3d(net, nb_filter=64, filter_size=3, activation="relu", strides=1, padding="valid")
	net = tflearn.fully_connected(net, 1024, activation="relu")
	net = tflearn.fully_connected(net, 256, activation="relu")
	net = tflearn.fully_connected(net, 2, activation="softmax")
	net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', batch_size=bs)
	return net