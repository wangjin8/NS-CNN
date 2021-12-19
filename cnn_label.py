#/usr/bin/python3.6
#coding:utf-8

from __future__ import print_function
import tensorflow as tf
import numpy as np
import random,csv
import time
import os
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def next_batch(feature_list,label_list,size):
    feature_batch_temp=[]
    label_batch_temp=[]
    # 从feature_list中随机获取5个元素，作为一个片断返回
    f_list = random.sample(range(len(feature_list)), size)
    for i in f_list:
        feature_batch_temp.append(feature_list[i])
    for i in f_list:
        label_batch_temp.append(label_list[i])
    return feature_batch_temp,label_batch_temp

def weight_variable(shape,layer_name):
    #定义一个shape形状的weights张量
    with tf.name_scope(layer_name + '_Weights'):
        Weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1),name='W')
    tf.summary.histogram(layer_name + '_Weights', Weights)
    return Weights

def bias_variable(shape,layer_name):
    #定义一个shape形状的bias张量
    with tf.name_scope(layer_name + '_biases'):
        biases = tf.Variable(tf.constant(0.1, shape=shape),name='b')
    tf.summary.histogram(layer_name + '_biases', biases)
    return biases

def conv2d(x, W,layer_name):
    #卷积计算函数
    # stride [1, x步长, y步长, 1]
    # padding:SAME/FULL/VALID（边距处理方式），same使输出特征图和输入特征图形状相同
    with tf.name_scope(layer_name + '_h_conv2d'):
        h_conv2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv2d

def max_pool_2x1(x,layer_name):
    # max池化函数
    # ksize [1, x边长, y边长,1] 池化窗口大小
    # stride [1, x步长, y步长, 1]
    # padding:SAME/FULL/VALID（边距处理方式）
    with tf.name_scope(layer_name + '_h_pool'):
        h_pool = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')#2x2的池化窗口 步长为1
    return h_pool

def max_pool_2x2(x,layer_name):
    # max池化函数
    # ksize [1, x边长, y边长,1] 池化窗口大小
    # stride [1, x步长, y步长, 1]
    # padding:SAME/FULL/VALID（边距处理方式）
    with tf.name_scope(layer_name + '_h_pool'):
        h_pool = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')#填充，2x2的池化窗口 步长为2
    return h_pool

def load_data():
    global feature,feature0,feature1,feature2,feature3,feature3
    global label
    global feature_full
    global label_full
    feature=[]
    feature0=[]
    feature1=[]
    feature2=[]
    feature3=[]
    feature4=[]
    label=[]
    label0=[]
    label1=[]
    label2=[]
    label3=[]
    label4=[]
    feature_full=[]
    label_full=[]

    print(os.getcwd())

    # file_path = 'Data/kddcup.data_10_percent_corrected_handled-5-label.csv'
    file_path = 'Data/10_percent.csv'

    print( file_path)
    with (open(file_path,'r')) as data_from:
        csv_reader=csv.reader(data_from)
        for i in csv_reader:
            # print i
            if int(i[41]) == 0:
                label_list=[0]*5
                feature0.append(i[:36])
                label_list[int(i[41])]=1
                label0.append(label_list)
            elif int(i[41]) == 1:
                label_list=[0]*5
                feature1.append(i[:36])
                label_list[int(i[41])]=1
                label1.append(label_list)
            elif int(i[41]) == 2:
                label_list=[0]*5
                feature2.append(i[:36])
                label_list[int(i[41])]=1
                label2.append(label_list)
            elif int(i[41]) == 3:
                label_list=[0]*5
                feature3.append(i[:36])
                label_list[int(i[41])]=1
                label3.append(label_list)
            else:
                label_list=[0]*5
                feature4.append(i[:36])
                label_list[int(i[41])]=1
                label4.append(label_list)
#填充原数据
        # for i in csv_reader:
        #     # print i
        #     if int(i[41]) == 0:
        #         label_list=[0]*5
        #         feature0.append(i[:40])
        #         # feature0.extend((0,0,0,0,0,0,0,0,0))
        #         addation = ('0', '0', '0', '0', '0', '0', '0', '0', '0')
        #         feature0.extend(addation)
        #         label_list[int(i[41])]=1
        #         label0.append(label_list)
        #     elif int(i[41]) == 1:
        #         label_list=[0]*5
        #         feature1.append(i[:40])
        #         addation = ('0', '0', '0', '0', '0', '0', '0', '0', '0')
        #         feature1.extend(addation)
        #         label_list[int(i[41])]=1
        #         label1.append(label_list)
        #     elif int(i[41]) == 2:
        #         label_list=[0]*5
        #         feature2.append(i[:40])
        #         addation = ('0', '0', '0', '0', '0', '0', '0', '0', '0')
        #         feature2.extend(addation)
        #         label_list[int(i[41])]=1
        #         label2.append(label_list)
        #     elif int(i[41]) == 3:
        #         label_list=[0]*5
        #         feature3.append(i[:40])
        #         addation = ('0', '0', '0', '0', '0', '0', '0', '0', '0')
        #         feature3.extend(addation)
        #         label_list[int(i[41])]=1
        #         label3.append(label_list)
        #     else:
        #         label_list=[0]*5
        #         feature4.append(i[:40])
        #         addation = ('0', '0', '0', '0', '0', '0', '0', '0', '0')
        #         feature4.extend(addation)
        #         label_list[int(i[41])]=1
        #         label4.append(label_list)

        #
        k=0.7#训练集占比
        #训练集初始化
        #不平衡数据集的过采样
        # kk =0.5 # dos数据的缩减
        # m0=2
        # m2=47
        # m3=6520
        # m4=1250
        kk =1 # dos数据的缩减
        m0=1
        m2=1
        m3=1
        m4=1
        feature_full=m0*feature0[0:int(k*len(label0))]+feature1[0:int(kk*k*len(label1))]+m2*feature2[0:int(k*len(label2))]+m3*feature3[0:int(k*len(label3))]+m4*feature4[0:int(k*len(label4))]
        label_full=m0*label0[0:int(k*len(label0))]+label1[0:int(kk*k*len(label1))]+m2*label2[0:int(k*len(label2))]+m3*label3[0:int(k*len(label3))]+m4*label4[0:int(k*len(label4))]
        #验证集初始化
        feature=feature0[int(k*len(label0))+1:]+feature1[int(k*len(label1))+1:]+feature2[int(k*len(label2))+1:]+feature3[int(k*len(label3))+1:]+feature4[int(k*len(label4))+1:]
        label=label0[int(k*len(label0))+1:]+label1[int(k*len(label1))+1:]+label2[int(k*len(label2))+1:]+label3[int(k*len(label3))+1:]+label4[int(k*len(label4))+1:]

        # # print label
        # print feature

if __name__  == '__main__':
    global feature
    global label
    global feature_full
    global label_full
    start_time = time.perf_counter()
    # load数据
    load_data()

    feature_test = feature
    feature_train =feature_full
    label_test = label
    #label_test_full = label_full
    label_train = label_full
    # 定义用以输入的palceholder
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 36],name='pic_data') # 6x6
        ys = tf.placeholder(tf.float32, [None, 5],name='pic_label')
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        x_image = tf.reshape(xs, [-1, 6, 6, 1])    # -1表示不约束这个位置 1表示信道1（灰度图仅有一个信道）

    ## 第一个卷积层 ， 下面是在另外一个命名空间来定义变量的
    with tf.name_scope('conv1_layer'):
        W_conv1 = weight_variable([3,3,1,32],layer_name='conv1')    # 卷积窗 3x3, 输入厚度 1, 输出厚度 32
        b_conv1 = bias_variable([32],layer_name='conv1')
        # h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1,layer_name='conv1') + b_conv1)    # 输出大小： 6x6x32
        # h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1, layer_name='conv1') + b_conv1)  # 输出大小： 6x6x32
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, layer_name='conv1') + b_conv1)  # 输出大小： 6x6x32

    # 第一个池化层
    with tf.name_scope('pool1_layer'):
        h_pool1 = max_pool_2x2(h_conv1,layer_name='pool1')    #2x2的池化窗口 步长为1,输出大小： 3x3x32

    ## 第二个卷积层 ， 下面是在另外一个命名空间来定义变量的
    with tf.name_scope('conv2_layer'):
        W_conv2 = weight_variable([1,1,32,32],layer_name='conv2')    # 卷积窗 1x1, 输入厚度 32, 输出厚度 32
        b_conv2 = bias_variable([32],layer_name='conv2')
        #h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2,layer_name='conv2') + b_conv2)    # 输出大小： 3x3x32
        # h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2, layer_name='conv2') + b_conv2)  # 输出大小： 3x3x32
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, layer_name='conv2') + b_conv2)  # 输出大小： 3x3x32

    # 第二个池化层
    with tf.name_scope('pool2_layer'):
        h_pool2 = max_pool_2x2(h_conv2,layer_name='pool2')  #2x2的池化窗口 步长为2,输出大小： 2x2x32

    ## 第一个全连接层 ##
    with tf.name_scope('fc1_layer'):
        W_fc1 = weight_variable([2*2*32,1024],layer_name='fc1')
        b_fc1 = bias_variable([1024],layer_name='fc1')
        with tf.name_scope('reshape'):
            h_pool2_flat = tf.reshape(h_pool2, [-1,2*2*32])
        with tf.name_scope('relu'):
            #h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            #h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # 带有dropout随机失活
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## 第二个全连接层 ##
    with tf.name_scope('fc2_layer'):
        W_fc2 = weight_variable([1024, 5],layer_name='fc2')
        b_fc2 = bias_variable([5],layer_name='fc2')
        with tf.name_scope('softmax'):
            prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    b = tf.constant(value=1, dtype=tf.float32)
    prediction_eval = tf.multiply(prediction, b, name='prediction_eval')
        # 计算loss/cost
    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))       # loss tf.clip_by_value(prediction,1e-10,1.0)
    tf.summary.scalar('loss',cross_entropy)
    # 计算accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    # 使用Adam优化器来实现梯度最速下降
    with tf.name_scope('train'):
        # 实例化     minimize更新参数
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.Session() as sess:
        # 初始化所有张量
        sess.run(tf.global_variables_initializer())
        # 将神经网络结构画出来
        # / 显示tensorboard
        writer = tf.summary.FileWriter("cnn-label_logs/", sess.graph)
        # 分别分出训练和评估时标记值的变化
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("cnn-label_logs/train", sess.graph)
        test_writer = tf.summary.FileWriter("cnn-label_logs/test", sess.graph)
        saver = tf.train.Saver(max_to_keep=1)#保存全部已训练模型

        pr=[]

        for step in range(301):

            feature_train_batch, label_train_batch =next_batch(feature_train, label_train,10000)  # 随机梯度下降训练，每次选大小为1000的batch
            feature_test_batch, label_test_batch = next_batch(feature_test, label_test,10000)  # 随机梯度下降训练，每次选大小为1000的batch
            sess.run(train_step, feed_dict={xs: feature_train_batch, ys: label_train_batch, keep_prob: 0.5})
             # 每25迭代次数将网络的状况体现在summary里

            if step % 25 == 0:
                train_writer.add_summary(sess.run(merged, feed_dict={xs: feature_train_batch, ys: label_train_batch, keep_prob: 1}), step)
                test_writer.add_summary(sess.run(merged, feed_dict={xs: feature_test, ys: label_test, keep_prob: 1}), step)
                if step == 25:
                    print("训练轮数 预测数据  真实数据  训练准确率 测试准测率")
                print(step,
                      sess.run(tf.argmax(prediction, 1)[7:27], feed_dict={xs: feature_test, ys: label_test, keep_prob: 1}),
                      sess.run(tf.argmax(ys, 1)[7:27], feed_dict={xs: feature_test, ys: label_test, keep_prob: 1}),
                      sess.run(accuracy, feed_dict={xs: feature_train_batch, ys: label_train_batch, keep_prob: 1}),
                      sess.run(accuracy, feed_dict={xs: feature_test, ys: label_test, keep_prob: 1}))
                pr.append(sess.run(accuracy, feed_dict={xs: feature_train_batch, ys: label_train_batch, keep_prob: 1}))
            # saver.save(sess, 'ckpt/scnn-label.ckpt')
            saver.save(sess, 'ckpt/scnn-label.ckpt', global_step=step) # 将训练次数存进名字
    end_time = time.perf_counter()
    print("train accuracy:", pr)

    temp = sys.stdout  # 记录当前输出指向，默认是consle
    with open("Result/train result.txt", "w") as f:
        sys.stdout = f  # 输出指向txt文件
        print("train accuracy:", pr)
        print()
        print("Running time:", (end_time - start_time))
        sys.stdout = temp  # 输出重定向回consle


    print("Running time:", (end_time - start_time))  # 输出程序运行时间