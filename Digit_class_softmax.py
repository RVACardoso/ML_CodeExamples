from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#get data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#initializing parameters for the model
batch_size = 100
eta = 0.001
num_iter = 35

x = tf.placeholder(tf.float32, shape=[None, 784])
target = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#initializing the model and define cost function
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(target * tf.log(y), reduction_indices=[1]))

#get accuracy of parameters
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#minimize cost with gradient descent
train_op = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy)

#start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(num_iter):
    batch_count = int(mnist.train.num_examples / batch_size)
for i in range(batch_count):
    batch_x, batch_y = mnist.train.next_batch(batch_size)

#run
sess.run([train_op], feed_dict={x: batch_x, target: batch_y})

#show accuracy
print('Epoch: {}'.format(epoch))
print("Accuracy: {}".format(accuracy.eval(feed_dict={x: mnist.test.images, target: mnist.test.labels}, session=sess)))
print("Model Execution Complete")