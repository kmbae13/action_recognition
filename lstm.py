from __future__ import print_function

import numpy as np
import tensorflow as tf
#import tf.nn.rnn_cell as rnn
import random
import collections
import time
import pdb
from tqdm import tqdm
from tqdm import trange
from PIL import Image
import glob
#import csv
import os
import re
import test_lstm
from tensorflow.contrib.slim.python.slim.nets import inception

rnn = tf.nn.rnn_cell
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

def select_train_test_data(trainlist):
    train = []
    test = []
    label = {}
    with open('./data/ucfTrainTestlist/classInd.txt','r') as f:
        contents = f.readlines()
    contents = [x.strip() for x in contents]
    contents = [x.split(' ') for x in contents]
    for l in contents:
        label.update({l[1]:l[0]})

    with open('./data/ucfTrainTestlist/trainlist0{}.txt'.format(trainlist), 'r') as f:
        contents = f.readlines()
    contents = [x.strip() for x in contents]
    #contents = [x.replace('/',' ').replace('.avi','') for x in contents]
    contents = [x.split(' ') for x in contents]
    train = contents
    #[train.append(x[1:]) for x in contents]

    with open('./data/ucfTrainTestlist/testlist0{}.txt'.format(trainlist), 'r') as f:
        contents = f.readlines()
    contents = [x.strip() for x in contents]
    contents = [x.replace('/',' ').replace('.avi','') for x in contents]
    contents = [x.split(' ') for x in contents]
    contents = [[x[1],label[x[0]]] for x in contents]
    [test.append(x) for x in contents]

    train = np.asarray(train)
    test = np.asarray(test)
    return train, test, label

def select_split(trainlist):
    label = {}
    with open('./data/ucfTrainTestlist/classInd.txt','r') as f:
        contents = f.readlines()
    contents = [x.strip() for x in contents]
    contents = [x.split(' ') for x in contents]
    for l in contents:
        label.update({l[1]:l[0]})
    with open('./data/ucfTrainTestlist/trainlist0{}.txt'.format(trainlist), 'r') as f:
        train = np.loadtxt(f)
    with open('./data/ucfTrainTestlist/testlist0{}.txt'.format(trainlist), 'r') as f:
        test = np.loadtxt(f)
    return train, test, label

def read_data(fname_list):
    out_data = []
    out_data_y = []
    #pdb.set_trace()
    for fname_tmp in fname_list:
        fname = glob.glob( './data/sequences_npy/' + str(fname_tmp[0].split('/')[-1].split('.')[0]) + '*.npz')
        if fname == []:
            continue
        with open( fname[0], 'r') as f:
            data = np.load(f)
        #data = [x.strip() for x in data]
        #data = [x.split(' ') for x in data]
        #data = np.array(data)
        out_data.append(data)
        tmp = np.zeros(101)
        tmp[int(fname_tmp[1])-1] = 1
        out_data_y.append(tmp)
    #data = np.stack(out_data)
    idx_in = [xx.shape[0] for xx in out_data]
    data = np.concatenate(out_data,0)
    return data, idx_in, np.asarray(out_data_y)

def read_images(list_set):
    read_im = []
    label = []
    for i,jj in list_set:
        read_vi = []
        length = len(glob.glob('data/UCF101_frame_10/'+i+'/*'))
        for j in range(length):
            im = Image.open('data/UCF101_frame/'+i+'/{:05d}.jpg'.format(j+1))
            im = im.resize((299,299))
            read_vi.append(np.array(im))
        read_im.append(np.stack(read_vi))
        tmp = np.zeros(101)
        tmp[int(jj) - 1] = 1
        label.append(tmp)
    input_in = np.concatenate(read_im,0)
    idx_in = [xx.shape[0] for xx in read_im]

    return input_in, idx_in, np.stack(label)


def calculate_feature_score(input_features, W1, b1, W2, b2, W3, b3):
    hidden_layer = 1024
    with tf.variable_scope("cal_score", reuse=True):

        act1 = tf.tanh(tf.add(tf.matmul(input_features,W1),b1))
        act3 = tf.tanh(tf.add(tf.matmul(act1,W2),b2))
        act2 = tf.sigmoid(tf.add(tf.matmul(act3,W3),b3))

    return act2

def RNN(x, weights, biases, output_keep_prob):
    # x.shape = [batch_size,Sequence length, feature_size]

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    #x = [tf.reshape(xx,[-1, feature_size]) for xx in tf.split(x,n_input,1)]
    #x = tf.stack(x,1)
    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.LSTMCell(n_hidden)

    # generate prediction
    rnn_cell_dropout = rnn.DropoutWrapper(rnn_cell, output_keep_prob)

    output_rnn, states = tf.nn.dynamic_rnn(rnn_cell_dropout, x, dtype=tf.float32)
    #pdb.set_trace()
    #
    # outputs = tf.nn.relu(tf.matmul(output_rnn[-1],weights['hidden']) + biases['hidden'])
    tmp = (tf.matmul(output_rnn[-1],weights['hidden']) + biases['hidden'])
    # there are n_input outputs but
    # we only want the last output
    # tmp = tf.reduce_mean(output_rnn,1)
    outputs = (tf.matmul(tmp,weights['hidden']) + biases['hidden'])
    #tmp = outputs[-1]
    return tf.matmul(outputs, weights['out']) + biases['out']

start_time = time.time()

# Parameters
logs_path = './logs_rnn'
n_hidden = 2048
t = time.asctime(time.localtime())
learning_rate = 0.001
training_iters = 150
display_step = 10
n_input = 10 # number of important frames to select
feature_size = 2048
output_size = 101
hidden_size = 256
batch_size = 32
n_hidden = 256
channel = 3
hidden_layer = 2048 # number of hidden units for score calculation function
hidden_layer2 = 512

# tf Graph input
x = tf.placeholder("float", [None, 299, 299, channel], name="x")
ind = tf.placeholder(tf.int32, [batch_size], name="index")
y = tf.placeholder("float", [None, output_size], name="y")
prob = tf.placeholder_with_default(1.0, shape=(), name="Dropout_keep_probability")
input_feature = tf.placeholder(tf.float32, shape=(None, feature_size))

# Variables

W1 = tf.Variable(tf.Variable(tf.random_normal([2048, hidden_layer])),name="W1")
b1 = tf.Variable(tf.Variable(tf.random_normal([hidden_layer])),name='b1')
W2 = tf.Variable(tf.Variable(tf.random_normal([hidden_layer,hidden_layer2])),name="W2")
b2 = tf.Variable(tf.Variable(tf.random_normal([hidden_layer2])),name='b2')
W3 = tf.Variable(tf.Variable(tf.random_normal([hidden_layer2,1])),name="W3")
b3 = tf.Variable(tf.Variable(tf.random_normal([1])),name='b3')

#pdb.set_trace()
slim = tf.contrib.slim

# Inception model
with slim.arg_scope(inception.inception_v3_arg_scope()):
    _, end_points = inception.inception_v3(x, num_classes=1001, is_training=False)

# Selecting top n number of frames
output_feature = tf.squeeze(end_points['PreLogits'],[1,2])
#feature vector shape=(batch_size,num of frames,feature vector size)
# bb = tf.split(output_feature, ind)
bb = tf.split(input_feature, ind)
out = []
for i in bb:
    aa = calculate_feature_score(i,W1,b1,W2,b2,W3,b3)  #importance score shape=(batch_size,num of frames)
    sorted_a, indices = tf.nn.top_k(tf.transpose(aa), n_input)
    indices = tf.transpose(tf.transpose(indices)[::-1])
    #shape_a = tf.shape(aa)
    #auxiliary_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(shape_a[:(aa.get_shape().ndims - 1)]) + [n_input])], indexing='ij')
    #sorted_b = tf.gather_nd(bb, tf.stack(auxiliary_indices[:-1] + [indices], axis=-1))# selected k number of important feature vector
    sorted_b = tf.gather(i,tf.squeeze(indices,0))[::-1]
    out.append(sorted_b)
out = tf.stack(out,0)
out = tf.transpose(out,[1,0,2])
# RNN output node weights and biases
with tf.variable_scope('LSTM'):
    weights = {'out': tf.Variable(tf.random_normal([hidden_size, output_size], name="w_out"))
     , 'hidden':tf.Variable(tf.random_normal([n_hidden, hidden_size], name="w_hidden"))}
    biases = {'out': tf.Variable(tf.random_normal([output_size], name="b_out"))
     , 'hidden': tf.Variable(tf.random_normal([hidden_size], name="b_hidden"))}

pred = RNN(out, weights, biases, prob)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred), name="cost")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer").minimize(cost)
tf.summary.scalar('Loss', cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
tf.summary.scalar('Accuracy', accuracy)

# tf.summary.scalar('Accuracy', accuracy)
# tf.add_to_collection("cost",cost)
# tf.add_to_collection("accuracy",accuracy)

# Merge summary
merge = tf.summary.merge_all()
#writer = tf.summary.FileWriter('{}/{}/{}/train/'.format(logs_path, n_hidden, t))
#writer_test = tf.summary.FileWriter('{}/{}/{}/test/'.format(logs_path, n_hidden, t))

# Initializing the variables
init = tf.global_variables_initializer()
# init = tf.initialize_all_variables()

# Defining global step
# global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0),trainable=False)

# Saving model
saver = tf.train.Saver()

# input_x_test, out_y_test = read_data(test_set)
# Launch the graph

for i in range(3):

    writer = tf.summary.FileWriter('{}/{}/{}/split0{}/train/'.format(logs_path, n_hidden, t,i))
    writer_test = tf.summary.FileWriter('{}/{}/{}/split0{}/test/'.format(logs_path, n_hidden, t,i))

    train_set, test_set, label = select_train_test_data(i+1)
    #input_x_test, out_y_test = read_data(test_set)
    with tf.Session() as session:
        session.run(init)
        idx = 0
        acc_total = 0
        loss_total = 0
        writer.add_graph(session.graph)
        # train_idx = np.random.permutation(len(train_set))
        restore = 0
        #saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionV3'))
        #saver1.restore(session,'./inception-v3/inception_v3.ckpt')
        if restore != 0:
            t = ''
            saver.restore(session, './saved_model/{}/{}/LSTM_var-{}'.format(n_hidden, t, restore))
            print('Model restored!')

        for epoch in range(restore, training_iters):
            # Generate a minibatch. Add some randomness on selection process.
            idx = 0
            train_idx = np.random.permutation(train_set)
            train_idx = np.split(train_idx, np.arange(batch_size, len(train_idx), batch_size))
            for path in tqdm(train_idx[:-1]):
                features, read_idx, out_y = read_data(path)
                # input_x, read_idx, out_y = read_images(path)
                # input_x = np.reshape(input_x, [-1, n_input, feature_size])
                # out_y = np.zeros(input_x.shape, dtype=float)
                # out_y[int(label[train_idx[idx][0]]) - 1] = 1.0
                # out_y = np.reshape(out_y,[1,-1])
                '''features = []
                for rr in np.split(input_x, read_idx[0:-1]):
                    features.append(session.run(output_feature,\
                             feed_dict={x:rr}))
                features = np.concatenate(features,0)'''
                _ = session.run([optimizer], \
                                       feed_dict={prob: 0.5, ind: read_idx, input_feature: features, y: out_y})

                if idx % display_step == 0:
                    result, acc, loss = session.run([merge, accuracy, cost], \
                                        feed_dict={input_feature: features, y: out_y, ind: read_idx})
                    writer.add_summary(result, epoch*len(train_idx)+idx)

                idx += 1

                if idx % display_step == 0:
                    print("Iter= " + '{}/{}'.format(epoch, training_iters) + ", Loss= " + \
                          "{:.6f}".format(loss) + ", Accuracy= " + \
                          "{:.2f}%".format(100 * acc) + " Saved location : " + t)
                    # step += 1

            if not os.path.isdir('./saved_model/{}/{}/split0{}'.format(n_hidden, t, i)):
                os.makedirs('./saved_model/{}/{}/split0{}'.format(n_hidden, t, i))
            saver.save(session, './saved_model/{}/{}/split0{}/LSTM_var'.format(n_hidden, t, i), global_step=epoch)
            print("Model saved")
            max_idx = idx
            test_idx = np.split(test_set, np.arange(batch_size, len(test_set), batch_size))
            acc_cum = 0
            loss_cum = 0
            for path in tqdm(test_idx[:-1]):
                input_x_test, ii, out_y_test = read_data(path)
                result, acc, loss, onehot_pred = session.run([merge, accuracy, cost, pred], \
                                                 feed_dict={input_feature: input_x_test, ind: ii, y: out_y_test})
                acc_cum+=acc/batch_size
                loss_cum+=loss/batch_size
            writer_test.add_summary(result, epoch*len(train_idx)+idx)
            print("Iter= " + '{}/{}'.format(epoch, training_iters) + ",Test loss= " + \
                          "{:.6f}".format(loss) + ", Test accuracy= " + \
                          "{:.2f}%".format(100 * acc) + " Saved location : " + t)
            #re_acc, re_cost = test_lstm.test(t, epoch, n_hidden, test_set)
        print("Optimization Finished!")
        print("Elapsed time: ", elapsed(time.time() - start_time))

        acc_total = 0
        loss_total = 0

    #re_acc, re_cost = test_lstm.test(t, 9, n_hidden, test_set)
    #result_acc.append(re_acc)
    #result_cost.append(re_cost)


