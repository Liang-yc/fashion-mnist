
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import timeit
import tensorflow as tf
import numpy
from  six.moves import xrange
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from PIL import Image
# import mnist_data

FLAGS = None
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
SEED =0
BATCH_SIZE = 100
MAX_STEPS=2400

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == labels) /
        predictions.shape[0])


def pro_processing(image,i):
    img=Image.fromarray(numpy.reshape(image,(28,28))*255)
    img.show()
    # img=img.transpose(Image.FLIP_LEFT_RIGHT)
    img=img.rotate((i%4)*90)
    image=[numpy.asarray(img)]
    image=image[:,numpy.newaxis]
    return image

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def main(_):
  numpy.random.seed(1280)
  # Import data
  mnist = read_data_sets(FLAGS.data_dir, reshape=False,validation_size=0, source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

  #data argument
  # train_data, train_labels, _,_,test_data,test_labels = mnist_data.prepare_MNIST_data(False)
  # for i in xrange(mnist.train.num_examples):
  #     mnist.train.images[i]=pro_processing(mnist.train.images[i],i)

  # image=mnist.train.images[1000].reshape(28,28)
  # im=Image.fromarray(image*255)
  # im.show()
  # Create the model

  with tf.Graph().as_default():
      # This is where training samples and labels are fed to the graph.
      # These placeholder nodes will be fed a batch of training data at each
      # training step using the {feed_dict} argument to the Run() call below.
      # is_train=tf.placeholder(tf.bool,name='is_train')
      data = tf.placeholder(
          tf.float32,
          shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name='Input')

      labels = tf.placeholder(tf.int64, shape=(None,), name='Label')

      tst = tf.placeholder(tf.bool)
      is_train=tf.placeholder(tf.bool)
      iter = tf.placeholder(tf.int32)
      keep_prob = tf.placeholder(tf.float32)
      # The variables below hold all the trainable weights.
      with tf.name_scope('conv1'):
          conv1_weights = tf.Variable(tf.truncated_normal(
              [1, 3, NUM_CHANNELS, 64],
              stddev=0.1,
              seed=SEED,
              dtype=tf.float32))
          conv1_biases = tf.Variable(tf.zeros([64], dtype=tf.float32))
          conv = tf.nn.conv2d(data,
                              conv1_weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME')
          # convbn, update_ema1 = batchnorm(conv, tst, iter, conv1_biases , convolutional=True)
          conv=tf.nn.bias_add(conv, conv1_biases)
          convbn=tf.layers.batch_normalization(conv,training=is_train)
          relu=tf.nn.relu(convbn)
          # conv_bn = batch_norm(conv, 64, is_train)
          # relu = tf.nn.relu(tf.nn.bias_add(conv_bn, conv1_biases))
          pool = tf.nn.max_pool(relu,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 1, 1, 1],
                                padding='SAME')

      with tf.name_scope('conv2'):
          conv2_weights = tf.Variable(tf.truncated_normal(
              [3, 1, 64, 64],
              stddev=0.1,
              seed=SEED,
              dtype=tf.float32))
          conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
          conv = tf.nn.conv2d(pool,
                              conv2_weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME')
          conv = tf.nn.bias_add(conv, conv2_biases)
          # convbn, update_ema2 = batchnorm(conv, tst, iter, conv2_biases , convolutional=True)
          convbn=tf.layers.batch_normalization(conv,training=is_train)
          relu=tf.nn.relu(convbn)
          # conv_bn = batch_norm(conv, 64, is_train)
          # relu = tf.nn.relu(tf.nn.bias_add(conv_bn, conv2_biases))
          pool = tf.nn.max_pool(relu,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 1, 1, 1],
                                padding='SAME')

      with tf.name_scope('conv3'):
          conv3_weights = tf.Variable(tf.truncated_normal(
              [1, 3, 64, 64],
              stddev=0.1,
              seed=SEED,
              dtype=tf.float32))
          conv3_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
          conv = tf.nn.conv2d(pool,
                              conv3_weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME')
          conv = tf.nn.bias_add(conv, conv3_biases)
          convbn=tf.layers.batch_normalization(conv,training=is_train)
          # convbn, update_ema3 = batchnorm(conv, tst, iter, conv3_biases , convolutional=True)
          relu=tf.nn.relu(convbn)
          # conv_bn = batch_norm(conv, 64, is_train)
          # relu = tf.nn.relu(tf.nn.bias_add(conv_bn, conv3_biases))
          pool = tf.nn.max_pool(relu,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
      with tf.name_scope('conv4'):
          conv4_weights = tf.Variable(tf.truncated_normal(
              [3, 1, 64, 64],
              stddev=0.1,
              seed=SEED,
              dtype=tf.float32))
          conv4_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
          conv = tf.nn.conv2d(pool,
                              conv4_weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME')
          conv = tf.nn.bias_add(conv, conv4_biases)
          convbn=tf.layers.batch_normalization(conv,training=is_train)
          # convbn, update_ema4 = batchnorm(conv, tst, iter, conv4_biases , convolutional=True)
          relu=tf.nn.relu(convbn)
          # conv_bn = batch_norm(conv, 64, is_train)
          # relu = tf.nn.relu(tf.nn.bias_add(conv_bn, conv4_biases))
          pool = tf.nn.max_pool(relu,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
      with tf.name_scope('conv5'):
          conv5_weights = tf.Variable(tf.truncated_normal(
              [3, 3, 64, 64],
              stddev=0.1,
              seed=SEED,
              dtype=tf.float32))
          conv5_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
          conv = tf.nn.conv2d(pool,
                              conv5_weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME')
          conv = tf.nn.bias_add(conv, conv5_biases)
          convbn=tf.layers.batch_normalization(conv,training=is_train)
          # convbn, update_ema5 = batchnorm(conv, tst, iter, conv5_biases , convolutional=True)
          relu=tf.nn.relu(convbn)
          # conv_bn = batch_norm(conv, 64, is_train)
          # relu = tf.nn.relu(tf.nn.bias_add(conv_bn, conv5_biases))
          pool = tf.nn.max_pool(relu,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 1, 1, 1],
                                padding='SAME')
      #reshape
      pool_shape = pool.get_shape().as_list()
      reshape = tf.reshape(
          pool,
          [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])

      with tf.name_scope('fc1'):
          fc1_weights = tf.Variable(  # fully connected, depth 512.
              tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                                  stddev=0.1,
                                  seed=SEED,
                                  dtype=tf.float32))
          fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
          fc1=tf.add(tf.matmul(reshape, fc1_weights),fc1_biases)
          fc1=tf.layers.batch_normalization(fc1,training=is_train)
          fc1 = tf.nn.relu(fc1)
      # with tf.name_scope('dropout'):
      #     drop = tf.nn.dropout(fc1, keep_prob)
      with tf.name_scope('fc2'):
          fc2_weights = tf.Variable(tf.truncated_normal([512, 512],
                                                        stddev=0.1,
                                                        seed=SEED,
                                                        dtype=tf.float32))
          fc2_biases = tf.Variable(tf.constant(
              0.1, shape=[512], dtype=tf.float32))
          fc2 = tf.add(tf.matmul(fc1, fc2_weights), fc2_biases)
          # fc2,update_em6=batchnorm(tf.matmul(drop,fc2_weights),tst,iter,fc2_biases)
          fc2=tf.layers.batch_normalization(fc2,training=is_train)
          fc2=tf.nn.relu(fc2)
      with tf.name_scope('fc3'):
          fc3_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                        stddev=0.1,
                                                        seed=SEED,
                                                        dtype=tf.float32))
          fc3_biases = tf.Variable(tf.constant(
              0.1, shape=[NUM_LABELS], dtype=tf.float32))
          fc3_out = tf.add(tf.matmul(fc2, fc3_weights), fc3_biases, name="Output")
          # fc2_out=tf.nn.softmax(fc2_out)
      # update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4,update_ema5,update_em6)
      # Training computation: logits + cross-entropy loss.

      # update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)
      with tf.name_scope('Train'):
          loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=labels, logits=fc3_out))
          # L2 regularization for the fully connected parameters.
          regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                          tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) +
                          tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases))
          # Add the regularization term to the loss.
          loss += 5e-4 * regularizers

          extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          with tf.control_dependencies(extra_update_ops):
              optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

      saver = tf.train.Saver()
      sess = tf.Session()
      # summary = tf.summary.merge_all()
      # summary_writer = tf.summary.FileWriter("SaveFiles", graph=tf.get_default_graph())
      sess.run(tf.global_variables_initializer())
      # export Graph
      # tf.train.write_graph(sess.graph_def, "SaveFiles", "Graph.pb")

      with tf.Session() as sess:
          # Run all the initializers to prepare the trainable parameters.
          tf.global_variables_initializer().run()
          print('Initialized!')
          # Loop through training steps.
          start_epoch = 0
          # saver.restore(sess,"C:/Users/lenovo/Desktop/fashion-mnist-master/model_bn/model.ckpt-2400")
          checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
          if checkpoint:
              saver.restore(sess, checkpoint)
              print("## restore from the checkpoint {0}".format(checkpoint))
              start_epoch += int(checkpoint.split('-')[-1])
          print('start training')
          for step in range(start_epoch, MAX_STEPS):
              batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
              # offset = (step * BATCH_SIZE) % (len(train_labels))
              # batch_xs = train_data[offset:(offset + BATCH_SIZE)]
              # batch_ys = train_labels[offset:(offset + BATCH_SIZE)]
              # Run the optimizer to update weights.
              loss_value,_=sess.run([loss, optimizer],
                              feed_dict = {data: batch_xs,
                                           labels: batch_ys,
                                           # is_train:True,
                                           is_train: True})


              # Save Model
              if step %100 == 0:
                  print('Step: %d: loss: %.6f.' % (step, loss_value))
                  sys.stdout.flush()
                  checkpoint_file = os.path.join(FLAGS.model_dir, 'model.ckpt')
                  saver.save(sess, checkpoint_file, global_step=step)
                  # print('shuffle training data')
                  # perm = numpy.arange(len(train_data))
                  # numpy.random.shuffle(perm)
                  # train_data = train_data[perm]
                  # train_labels = train_labels[perm]

          print('saving trained model')
          saver.save(sess, './trained_model/model.ckpt')
          print('start testing')
          size = mnist.test.images.shape[0]
          predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
          # size=len(test_labels)
          # predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
          av_time = 0
          for begin in xrange(0, size, BATCH_SIZE):
              end = begin + BATCH_SIZE
              batch_xs, batch_ys = mnist.test.next_batch(BATCH_SIZE)
              # offset = (begin * BATCH_SIZE) % (len(test_labels))
              # batch_xs = test_data[offset:(offset + BATCH_SIZE)]
              # batch_ys = test_labels[offset:(offset + BATCH_SIZE)]
              start_time = timeit.default_timer()
              predictions[begin:end, :] = sess.run(
                  fc3_out,
                  feed_dict={data: batch_xs,
                             labels: batch_ys,
                             # is_train:False,
                             is_train: False})
              elapsed = timeit.default_timer() - start_time
              av_time = av_time + elapsed

          print("Number of test samples: ", size)
          print("Average run time per sample: ", av_time / size)
          test_error = error_rate(predictions, mnist.test.labels)
          # test_error = error_rate(predictions, test_labels)
          print('Test error: %.1f%%' % test_error)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='./data/fashion',
                      help='Directory for storing input data')
  parser.add_argument('--model_dir', type=str, default='./model_bn',
                      help='Directory for storing trained model')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
