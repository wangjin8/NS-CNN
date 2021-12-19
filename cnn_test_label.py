# import tensorflow as tf
import numpy as np
import random,csv
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import sys

global feature
global clabel

def test(test_file):
    feature=[]
    clabel=[]
    # 统计五分类过程五大类数据的具体分类情况
    wtf = np.zeros((5, 5))
    #加载测试集和标签
    file_path =test_file
    with (open(file_path,'r')) as data_from:
        csv_reader=csv.reader(data_from)
        for i in csv_reader:
            # print i
            if i[41]!='5':
                feature.append(i[:36])
                correct_label=i[41]
                clabel.append(correct_label)
            else:
                print()
            #print(feature)

    # saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('ckpt/')  # 通过检查点文件锁定最新的模型
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('CNN Model Loading Success')
        else:
            print('No Checkpoint')
            #ii=0
        graph = tf.get_default_graph()
        xs = graph.get_tensor_by_name("inputs/pic_data:0")
        keep_prob = graph.get_tensor_by_name("inputs/keep_prob:0")
        logits = graph.get_tensor_by_name("prediction_eval:0")

        prediction = sess.run(logits, feed_dict={xs: feature,keep_prob: 1.0})  ##?

        print('Prediction Matrix of Test Data Set:')
        print(prediction)
        max_index = np.argmax(prediction, 1)
        print('Prediction Vector of Test Data Set:')
        print(max_index)
        m0 = max_index
        print('Size of Test Data Set:   ',m0.shape)
       

        for ii in range(len(feature)):
            
            k = int(clabel[ii])
            #print(m0[5])
            if m0[ii] in range(5):
                wtf[k][m0[ii]] += 1

    print('Type0:   Normal:',sum(wtf[0]),' ',wtf[0][0])
    print('         Accuracy:', wtf[0][0] / sum(wtf[0]))
    print('Type1:   Dos:',sum(wtf[1]),' ',wtf[1][1])
    print('         Accuracy:', wtf[1][1] / sum(wtf[1]))
    print('Type2:   Probing:',sum(wtf[2]),' ',wtf[2][2])
    print('         Accuracy:', wtf[2][2] / sum(wtf[2]))
    print('Type3:   R2L(Remote to Local):',sum(wtf[3]),' ',wtf[3][3])
    print('         Accuracy:', wtf[3][3] / sum(wtf[3]))
    print('Type4:   U2R(User to Root):',sum(wtf[4]),' ',wtf[4][4])
    print('         Accuracy:', wtf[4][4] / sum(wtf[4]))

    print('Confusion Matrix:')
    print(wtf)
    total=sum(wtf[1])+sum(wtf[2])+sum(wtf[3])+sum(wtf[4])

    ns = (wtf[1][1] * 0.35 + wtf[2][2] * 0.15  + wtf[3][3] * 0.3+wtf[4][4] * 0.2) / total

    print(ns)
    print()
    print("网络态势等级：")
    if (ns >= 0.6):
        print("差")
    else:
        if (ns > 0.3):
            print("中")
        else:
            if (ns > 0.1):
                print("良")
            else:
                print("优")


    temp = sys.stdout  # 记录当前输出指向，默认是consle

    with open("Result/test result.txt", "w") as f:
        sys.stdout = f  # 输出指向txt文件
        # print("filepath:", __file__,
        #       "\nfilename:", os.path.basename(__file__))


        print('总数据条数为 ', (sum(wtf[1]) + sum(wtf[2]) + sum(wtf[3]) + sum(wtf[4]) + sum(wtf[0])).astype(int))
        print()
        print('类型1：  正常数据总数:', sum(wtf[0]).astype(int))
        print('         识别正常数据数 ', wtf[0][0].astype(int))
        print('         识别准确率:', wtf[0][0] / sum(wtf[0]))
        print()
        print('类型2：  拒绝服务数据总数:', sum(wtf[1]).astype(int))
        print('         识别拒绝服务数据总数: ', wtf[1][1].astype(int))
        print('         识别准确率:', wtf[1][1] / sum(wtf[1]))
        print()
        print('类型3：  端口扫描数据总数:', sum(wtf[2]).astype(int))
        print('         识别端口扫描数据总数', wtf[2][2].astype(int))
        print('         识别准确率:', wtf[2][2] / sum(wtf[2]))
        print()
        print('类型4：  未授权的远程访问数据总数:', sum(wtf[3].astype(int)))
        print('         识别未授权的远程访问数据总数:',wtf[3][3].astype(int))
        print('         识别准确率:', wtf[3][3] / sum(wtf[3]))
        print()
        print('类型5：  本地未授权的特权访问数据总数:', sum(wtf[4].astype(int)))
        print('         识别本地未授权的特权访问数据总数: ', wtf[4][4].astype(int))
        print('         识别准确率:', wtf[4][4] / sum(wtf[4]))
        print()
        print()
    #定义评估公式
        #ns = (wtf[1][1] * 0.35+4031 * 0.15 +  1100 * 0.3+ 52 * 0.2 ) / total  # 1
        ns = (wtf[1][1] * 0.35 + 4013 * 0.15 + 15925 * 0.3 + 217 * 0.2) / total
        #ns = ( wtf[1][1] * 0.3 +3997 * 0.15 +11954 * 0.3+  204 * 0.2 ) / total  # 3

        # ns = ( wtf[1][1] * 0.3 + 4003 * 0.1+15925 * 0.4 + 217 * 0.2) / total
        #ns = (4031 * 0.5 + wtf[1][1] * 0.25 + 52 * 0.9 + 1100 * 0.8) / total  # 1
        # ns = (3997 * 0.5 + wtf[1][1] * 0.25 +11954 * 0.8+  204 * 0.9 ) / total  # 3
        # print(ns)
        if (ns >= 0.6):
            print("差")
        else:
            if (ns > 0.3):
                print("网络态势等级：中")
            else:
                if (ns > 0.1):
                    print("网络态势等级：良")
                else:
                    print("网络态势等级：优")
        sys.stdout = temp  # 输出重定向回consle
        # print(f.readlines())  # 将记录在文件中的结果输出到屏幕
    return wtf

if __name__ == '__main__':
    start_time=time.perf_counter()
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('ckpt/')  # 通过检查点文件锁定最新的模型
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success')
        else:
            print('No checkpoint')

    # 获得几乎所有的operations相关的tensor
    # print('Get almost all operations related tensors before testing')
    # ops = [o for o in sess.graph.get_operations()]
    # for o in ops:
    #     print(o.name)
    # test('Data/kddcup.data_10_percent_corrected_handled-5-label.csv')
    # test('Data/kddcup.data_10_percent_corrected_handled_test.csv')
    #test('Data/corrected4.csv')
    test('Data/10_percent.csv')
    # test('Data/10_percent.csv')

    end_time=time.perf_counter()
    print("Running time:",(end_time-start_time))









