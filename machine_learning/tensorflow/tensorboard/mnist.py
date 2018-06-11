# Softmax Regression using tensorflow.
import tensorflow as tf

# Download the mnist data.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Input placeholder, 2-D tensor of floating-point nunbers.
# here None means that a dimension can be of any length.
X = tf.placeholder(tf.float32, [None, 784], name = 'X-input')

# New placeholder to input the correct answers.
Y = tf.placeholder(tf.float32, [None, 10], name = 'Y-input')

# Initialize both W and b as tensors full of zeros.
# Since we are going to learn W and b, it doesn't
# matter very much what they initial are.
W = tf.Variable(tf.zeros([784, 10]), name = 'Weight')
B = tf.Variable(tf.zeros([10]), name = 'Bias')

# Tensorboard histogram summary.
tf.summary.histogram('Weight', W)
tf.summary.histogram('Bias', B)

with tf.name_scope('Layer'):
    y = tf.nn.softmax(tf.matmul(X, W) + B)

with tf.name_scope('Cost'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y), \
                reduction_indices=[1]))
    # Tensorboard scalar summary.
    tf.summary.scalar('Cost', cross_entropy)

with tf.name_scope('Train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), \
               tf.argmax(Y, 1)), tf.float32))
    # Tensorboard scalar summary.
    tf.summary.scalar('Accuracy', accuracy)

with tf.Session() as sess:
    # Merge all summaries.
    writer = tf.summary.FileWriter('logs/mnist_logs', sess.graph)
    merged =  tf.summary.merge_all()

    tf.initialize_all_variables().run()

    # Training 1000 times, 100 for each loop.
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, summary = sess.run([train_step, merged], feed_dict={X: batch_xs, Y: batch_ys})
        # Write summary into files.
        writer.add_summary(summary, i)

    # Close summary writer.
    writer.close()
    print('Accuracy', accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
