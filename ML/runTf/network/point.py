import tflearn
import tensorflow as tf


def create(bs):
	def input_transform_net(point_cloud, K):
		num_point = point_cloud.get_shape()[1].value

		net = tflearn.conv_1d(point_cloud, nb_filter=64, filter_size=1, padding="valid", strides=1, activation="relu")
		net = tflearn.conv_1d(net, nb_filter=128, filter_size=1, padding="valid", strides=1, activation="relu")
		net = tflearn.conv_1d(net, nb_filter=256, filter_size=1, padding="valid", strides=1, activation="relu")
		net = tflearn.max_pool_1d(net, kernel_size=num_point, padding="valid")
		net = tflearn.fully_connected(net, 256, activation="relu")
		net = tflearn.fully_connected(net, 64, activation="relu")

		weights = tf.Variable(tf.zeros(shape=[64, K*K], dtype=tf.float32))
		biases = tf.Variable(tf.reshape(tf.eye(K, dtype=tf.float32), shape=[-1]))
		transform = tf.matmul(net, weights)
		transform = tf.nn.bias_add(transform, biases)
		transform = tf.reshape(transform, [-1, K, K])

		return transform


	net = tflearn.input_data(shape=[None, 57, 8])
	net_p = net[:,:,:3]
	net_m = net[:,:,3:]
	transform = input_transform_net(net_p, K=3)
	net = tf.matmul(net_p, transform)


	net = tflearn.conv_1d(net, nb_filter=32, filter_size=1, activation="relu", strides=1, padding="valid")
	net = tflearn.conv_1d(net, nb_filter=64, filter_size=1, activation="relu", strides=1, padding="valid")
	transform = input_transform_net(net, K=64)
	net = tf.matmul(net, transform)
	net = tf.concat((net, net_m), axis=2)
	
	net = tflearn.conv_1d(net, nb_filter=128, filter_size=1, activation="relu", strides=1, padding="valid")
	net = tflearn.conv_1d(net, nb_filter=256, filter_size=1, activation="relu", strides=1, padding="valid")
	net = tflearn.max_pool_1d(net, kernel_size=56, padding="valid")
	net = tflearn.fully_connected(net, 256, activation="relu")
	net = tflearn.fully_connected(net, 64, activation="relu")
	net = tflearn.fully_connected(net, 2, activation="softmax")
	net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', batch_size=bs)
	return net