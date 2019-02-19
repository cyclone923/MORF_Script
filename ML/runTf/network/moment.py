import tflearn


def create(n, bs):
	net = tflearn.input_data(shape=[None, n*4])
	net = tflearn.fully_connected(net, 256, activation="relu")
	net = tflearn.fully_connected(net, 128, activation="relu")
	net = tflearn.fully_connected(net, 64, activation="relu")
	net = tflearn.fully_connected(net, 2, activation="softmax")
	net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', batch_size=bs)
	return net