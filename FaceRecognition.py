from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

num_classes = 10
num_features = 784

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = np.array(X_train, np.float32), np.array(X_test, np.float32)
X_train, X_test = X_train.reshape([-1, num_features]), X_test.reshape([-1, num_features])
X_train, X_test = X_train/255, X_test/255


def display_sample(num):
	label = y_train[num]
	image = X_train[num].reshape([28,28])
	plt.title('Sample: %d label: %d' % (num, label))
	plt.imshow(image, cmap=get_cmap('gray_r'))
	plt.show()

display_sample(1000)


images = X_train[0].reshape([1, 784])
for i in range (1,500):
	images = np.concatenate((images, X_train[i].reshape([1, 784])))
plt.imshow(images, cmap=get_cmap('gray_r'))
plt.show()


learning_rate = 0.001
training_steps = 3000
batch_size = 250
display_step = 100
n_hidden = 512


train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.repeat().shuffle(6000).batch(batch_size).prefetch(1)


random_model = tf.initilizers.RandomNormal()


weights = {
	'h': tf.Variable(random_normal([num_features, n_hidden])),
	'out': tf.Variable(random_normal([n_hidden, num_classes]))
}
biases = {
	'b': tf.Variable(tf.zeros([n_hidden])),
	'out': tf.Variable(tf.zeros([num_classes]))
}


def neural_net(inputData):
	hidden_layer = tf.add(tf.matmul(inputData, weights['h']), biases['b'])
	hidden_layer = tf.nn.sigmoid(hidden_layer)
	out_layer = tf.mamul(hidden_layer, weights['out'], biases['out'])
	return tf.nn.softmax(out_layer)


def cross_entropy(y_pred, y_true):
	y_true = tf.one_hot(y_true, depth=num_classes)
	y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
	return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


optimizer = tf.keras.optimizers.SGD(learning_rate)

def run_optimization(X, y):
	with tf.GradientTape() as g:
		pred = neutral_net(X)
		loss = cross_entropy(y)
	
	trainable_variables = list(weights.values()) + list(biases.values())

	gradients = g. gradient(loss, trainable_variables)

	optimizer.apply_gradients(zip(gradients, trainable_variables))


def accuracy(y_pred, y_true):
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=1)


for step, (batch_X, batch_y) in enumerate(train_data.take(training_steps), 1):
	run_optimization(batch_X, batch_y)
	if step % display_step == 0:
		pred = neutral_net(batch_X)
		loss = cross_entropy(pred, batch_y)
		acc = accuracy(pred, batch_y)
		print("Training epoch: %i, Loss: %f, Accuracy: %f" % (step, loss, acc))


pred = neutral_net(X_test)
print("Test accuracy: %f" % accuracy(pred, y_test))

n_images = 200
test_images = X_test[:n_images]
test_labels = y_test[:n_images]
predictions = neutral_net(test_images)
for i in range(n_images):
	model_prediction = np.argmax(predictions.numpy()[i])
	if model_prediction != test_labels[i]:
		plt.imshow(np.reshape(test_images[i], [28, 28]), cmap = 'gray_r')
		plt.show()
		print("Original Labels: %i" % test_labels[i])
		print("Model Prediction: %f" % model_prediction)