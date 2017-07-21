import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Visualization
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Input data params
img_height = 28
img_width = 28
img_channels = 1
num_classes = 10

# Training params
learning_rate = 0.003
batch_size = 100
num_epochs = 60000 / batch_size

# Get the data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot = True)
X = tf.placeholder(tf.float32, [None, img_height * img_width])

# Placeholder for correct labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# Model
hl1_nodes = 200
hl2_nodes = 100
hl3_nodes = 60
hl4_nodes = 30

# Layer 1
W1 = tf.Variable(tf.truncated_normal([img_height * img_width, hl1_nodes], stddev = 0.1))
b1 = tf.Variable(tf.zeros([hl1_nodes]))
Y1 = tf.nn.relu(tf.matmul(X, W1) + b1)

# Layer 2
W2 = tf.Variable(tf.truncated_normal([hl1_nodes, hl2_nodes], stddev = 0.1))
b2 = tf.Variable(tf.zeros([hl2_nodes]))
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)

# Layer 3
W3 = tf.Variable(tf.truncated_normal([hl2_nodes, hl3_nodes], stddev = 0.1))
b3 = tf.Variable(tf.zeros([hl3_nodes]))
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)

# Layer 4
W4 = tf.Variable(tf.truncated_normal([hl3_nodes, hl4_nodes], stddev = 0.1))
b4 = tf.Variable(tf.zeros([hl4_nodes]))
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)

# Layer 5
W5 = tf.Variable(tf.truncated_normal([hl4_nodes, num_classes], stddev = 0.1))
b5 = tf.Variable(tf.zeros([num_classes]))
Y = tf.nn.softmax(tf.matmul(Y4, W5) + b5)

# Loss function - Cross Entropy
cross_entropy = - tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

# Create the session
sess = tf.Session()

# Initialize the variables
sess.run(tf.initialize_all_variables())

# Visualization
x_axis = []
training_accuracy_record = []
training_loss_record = []
testing_accuracy_record = []
testing_loss_record = []

# Run our Model!!
for i in range(num_epochs):
	# Load the input data
	batch_X, batch_Y = mnist_data.train.next_batch(batch_size)
	train_data = {X: batch_X, Y_: batch_Y}

	# Train
	sess.run(train_step, feed_dict = train_data)

	# Accuracy on training data
	acc, loss = sess.run([accuracy, cross_entropy], feed_dict = train_data)

	# Trying it out on test data
	test_data = {X: mnist_data.test.images, Y_: mnist_data.test.labels}

	# Accuracy on test data
	acc_test, loss_test = sess.run([accuracy, cross_entropy], feed_dict = test_data)
	
	x_axis.append(i)
	training_accuracy_record.append(acc)
	training_loss_record.append(loss)
	testing_accuracy_record.append(acc_test)
	testing_loss_record.append(loss_test)

	# Print stuff
	print (acc, loss, acc_test, loss_test)

plt.plot(x_axis, training_accuracy_record, 'r-', x_axis, testing_accuracy_record, 'b-')
plt.show()
plt.plot(x_axis, training_loss_record, 'r-', x_axis, testing_loss_record, 'b-')
plt.show()	