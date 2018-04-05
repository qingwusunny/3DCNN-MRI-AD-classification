# _*_coding:utf-8
#结构同train_mri1
import tensorflow as tf
import input_mri5
import numpy as np
import random
import os
import csv

keep_prob = 1  # 随机失活
n_epoch = 15001  # iteration number
weight_decay = 0.001
learning_rate = 0.00001
train_files=[]
txt_path = '/workspace/3dcnn/mycode_2017.7_3/result1.txt'
train_txt_path = '/workspace/3dcnn/mycode_2017.7_3/result1_train.txt'
ckpt_dir = '/workspace/3dcnn/mycode_2017.7_3/result_1_ckpt'
meta_path = '/workspace/3dcnn/mycode_2017.7_2/result_2_ckpt/model.ckpt-3000.meta'
model_path = '/workspace/3dcnn/mycode_2017.7_2/result_2_ckpt/model.ckpt-3000'
saver0 = tf.train.import_meta_graph(meta_path)
def weight_variable(value, name):
    return tf.Variable(value, name=name)

def bias_variable(value, name):
    return tf.Variable(value, name=name)


def conv3d(x_input, w, b):
    # strides[0]=strides[4]=1,strides[1],[2],[3]分别是滤波器在三个方向上滑动的步长
    return tf.nn.bias_add(
                           tf.nn.conv3d(x_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
                           b)


def conv3d_no_bias(x_input,w):
    return tf.nn.conv3d(x_input, w, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_3d(x_input, k):
    # ksize[0]=ksize[1]=1,[1],[2],[3]是max pool在三个方向上的尺寸
    return tf.nn.max_pool3d(x_input, ksize=[1, k, k, k, 1], strides=[1, k, k, k, 1], padding='SAME')

def avg_pool_3d(x_input, k):
    # ksize[0]=ksize[1]=1,[1],[2],[3]是max pool在三个方向上的尺寸
    return tf.nn.avg_pool3d(x_input, ksize=[1, k, k, k, 1], strides=[1, k, k, k, 1], padding='SAME')

def batch_normalization(x_input):
    x_shape = x_input.get_shape()
    axis = list(range(len(x_shape)-1))
    mean, variance = tf.nn.moments(x_input,axis)
    offset = bias_variable(x_shape[-1:], 'offset', value=0.0) 
    scale = bias_variable(x_shape[-1:], 'scale', value=1.0)
    return tf.nn.batch_normalization(x_input,mean,variance,offset,scale,0.001)


def cnn_layer(layer_name,x_input,weight_shape,bias_shape=None,flag_BN=None,flag_pool=None,pool_shape=None):
    # 没有batchnormalization, 或者没有pool时，不需要写相关参数
    # flag_BN='yes',需要做batchnormalization
    # 有BN时不需要bias
    # flag_pool='avg',avg_pool;flag_pool='max',max_pool
    # pool_shape, pooling的尺寸
    w_conv = weight_variable(weight_shape, 'wcnn')
    if flag_BN=='yes':
        h_conv = conv3d_no_bias(x_input,w_conv)
        h_bn_c = batch_normalization(h_conv)
    else:
        b_conv = bias_variable(bias_shape, 'bcnn')
        h_conv = conv3d(x_input, w_conv, b_conv)
        h_bn_c = h_conv
    h_relu = tf.nn.relu(h_bn_c)
    if flag_pool=='max':
        h_pool = max_pool_3d(h_relu,pool_shape)
    else:
        if flag_pool=='avg':
            h_pool = avg_pool_3d(h_relu,pool_shape)
        else:
            h_pool = h_relu

    return h_pool,w_conv


def fc_layer(layer_name,x_input,weight_shape,bias_shape=None,flag_BN=None,flag_relu=None,keep_prob=1, name=None):
    # 不需要做BN或relu或者drop时不需要写相关参数
    # 有BN时不需要bias
    # flag_BN='yes',需要做batchnormalization
    # flag_relu='yes',需要做relu
    # keep_prop>=1,不做drop;keep_prob<1,按所给的keep_prob做drop
    w_fc = weight_variable(weight_shape, 'wfc')
    if flag_BN=='yes':
        h_fc = tf.matmul(x_input, w_fc)
        h_bn = batch_normalization(h_fc)
    else:
        b_fc = bias_variable(bias_shape, 'bfc')
        h_fc = tf.matmul(x_input, w_fc) + b_fc
        h_bn = h_fc
    if flag_relu=='yes':
        h_relu = tf.nn.relu(h_bn,name=name)
    else:
        h_relu = h_fc
    if keep_prob>=1:
        f_drop = h_relu
    else:
        f_drop = tf.nn.dropout(h_relu, keep_prob)

    return  f_drop,w_fc
              

def acc(logits, labels):
    correct_pred = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=labels,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def run_training(fold,train_filenames,test_filenames):
    f = open(txt_path,'a')
    f.write('\r\n'+str(fold)+'th fold')
    f.close()
    f = open(train_txt_path,'a')
    f.write('\r\n'+str(fold)+'th fold')
    f.close()
    global train_files,learning_rate
    train_files = train_filenames
    images_placeholder = tf.placeholder(tf.float32, 
                                        shape=[None,input_mri5.width*input_mri5.height*input_mri5.depth],name='image')
    print(images_placeholder.name)                                                   
    labels_placeholder = tf.placeholder(tf.float32, shape=[None, input_mri5.num_class])
    x_image = tf.reshape(images_placeholder,[-1,61,73,61,1])
    print(labels_placeholder.name)
    #####################
     # convolution layer 1
    h_pool1,w_conv1 = cnn_layer(layer_name='cnnlayer1',x_input=x_image,weight_shape=[5, 5, 5, 1, 32],flag_BN='yes')
    #_x61x73x61x32
    h_pool1b,w_conv1b = cnn_layer(layer_name='cnnlayer1b',x_input=h_pool1,weight_shape=[3, 3, 3, 32, 32],bias_shape=[32],flag_pool='max',pool_shape=2)
    # _x31x37x31x32 
    
    # convolution layer 2
    h_pool2,w_conv2 = cnn_layer(layer_name='cnnlayer2',x_input=h_pool1b,weight_shape=[3, 3, 3, 32, 64],flag_BN='yes')
    h_pool2b,w_conv2b = cnn_layer(layer_name='cnnlayer2',x_input=h_pool2,weight_shape=[3, 3, 3, 64, 64],bias_shape=[64],flag_pool='max',pool_shape=4)
    # _x8x10x8x64 

    # convolution layer 3
    h_pool3,w_conv3 = cnn_layer(layer_name='cnnlayer3',x_input=h_pool2b,weight_shape=[3, 3, 3, 64, 128],flag_BN='yes',flag_pool='max',pool_shape=2)
    # _x4x5x4x128 
    h_pool4,w_conv4 = cnn_layer(layer_name='cnnlayer3',x_input=h_pool3,weight_shape=[3, 3, 3, 128, 256],bias_shape=[256],flag_pool='max',pool_shape=2)
    # _x2x3x2x128 

    # full connection layer 1
    h_flat = tf.reshape(h_pool4, [-1, 2 * 3 * 2 * 256])
    h_fc1,w_fc1 = fc_layer(layer_name='fclayer1',x_input=h_flat,weight_shape=[2 * 3 * 2 * 256, 1024],bias_shape=[1024],flag_relu='yes',name='h_fc1')
    h_fc2,w_fc2 = fc_layer(layer_name='fclayer2',x_input=h_fc1,weight_shape=[1024, 128],bias_shape=[128],flag_relu='yes',name='h_fc2')
    # full connection layer 2
    logit,w_logit = fc_layer(layer_name='fclayer3',x_input=h_fc2,weight_shape=[128,input_mri5.num_class],bias_shape=[input_mri5.num_class])
    out = tf.nn.softmax(logits=logit,name='out')
    ###############################

    l2_loss = tf.nn.l2_loss(w_conv1) + tf.nn.l2_loss(w_conv2) + tf.nn.l2_loss(w_conv3)+tf.nn.l2_loss(w_conv1b)+tf.nn.l2_loss(w_conv2b)+tf.nn.l2_loss(w_conv3)+tf.nn.l2_loss(w_conv4)+tf.nn.l2_loss(w_fc1)+tf.nn.l2_loss(w_fc2)+tf.nn.l2_loss(w_logit)
    accuracy = acc(logits=logit, labels=labels_placeholder)
    print('skssdnode:',accuracy.name)
    #tf.summary.scalar('accuracy',accuracy)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder,logits=logit))
    #tf.summary.scalar('loss',cross_entropy)
    all_loss = cross_entropy + weight_decay*l2_loss
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(all_loss)

    saver = tf.train.Saver(max_to_keep=None)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
    #with tf.Session() as sess:
        #merged = tf.summary.merge_all()
        #writer = tf.summary.FileWriter('tensorresult',sess.graph) 
        init = tf.global_variables_initializer()
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        for i in range(n_epoch):
            x_batch,y_batch,end = input_mri5.get_data_mri(train_filenames=train_files,batch_size=6)
            if end==len(train_filenames):
               random.shuffle(train_files)
            sess.run(train_step,feed_dict={images_placeholder:x_batch,labels_placeholder:y_batch})
            #if i%500==0 and i>4000:
                #saver.save(sess,ckpt_dir+'/model.ckpt',global_step=i)
            if i%200==0:
                [train_all_loss,train_loss,train_acc] = sess.run([all_loss,cross_entropy,accuracy],feed_dict={images_placeholder:x_batch,
                                                                 labels_placeholder:y_batch})
                f = open(train_txt_path,'a')
                f.write('\r\n'+'iteration:'+str(i)+'\t'+'train_all_loss:'+str(train_all_loss)+'\t'+'train_loss:'+str(train_loss)+'\t'+'accuracy:'+str(train_acc))
                f.close()
                if i%2000==0:
                    print('train:',i,train_all_loss,train_loss,train_acc)   
                #print('train:',i,train_loss,train_acc)
            #if i%50==0:
                #result = sess.run(merged,feed_dict={images_placeholder:x_batch,labels_placeholder:y_batch})
                #writer.add_summary(result,i)
            if i%5000==0:
                learning_rate=learning_rate/5.0
            if i%200==0:
                test_accuracy = []
                sensitivity = []
                specificity = []
                for test_index in range(len(test_filenames)):
                    x_test,y_test,_ = input_mri5.get_test_mri(test_filenames,batch_size=1)
                    one_acc = sess.run(accuracy, feed_dict={images_placeholder: x_test,
                                                    labels_placeholder: y_test})
                    #writer.add_summary(result1,i)
                    #print(test_out,y_test)
                    test_accuracy.append(one_acc)
                    if y_test[0][0]==1:
                        sensitivity.append(one_acc)
                    if y_test[0][1]==1:
                        specificity.append(one_acc)
                Sens = sum(sensitivity)/len(sensitivity)
                Spec = sum(specificity)/len(specificity)       
                test_ACC = sum(test_accuracy)/len(test_accuracy)
                f = open(txt_path,'a')
                f.write('\r\n'+'iteration:'+str(i)+'\t')
                f.write('accuracy:'+str(test_ACC)+'\t')
                f.write('sensitivity:'+str(Sens)+'\t')
                f.write('specificity:'+str(Spec))
                f.close()
                if test_ACC>0.88: #and (i%600==0):
                    saver.save(sess,ckpt_dir+'/model.ckpt',global_step=i)
                if i%2000==0:
                    print('test:', i, 'test_ACC:',test_ACC,'sensitivity:',Sens,'specificity:',Spec)
                    
def get_file():
    csv_reader=csv.reader(open('/workspace/3dcnn/mycode_2017.7_2/new_file3.csv'))
    data=[]
    label=[]
    filename=[]
    for row in csv_reader:
        filename.append(row[0])
        label.append(row[1])
    label = list(map(int,label))
    for i in range(len(label)):
        data.append([filename[i],label[i]])
    return data


def main(_):
    f0 = open(txt_path,'a')
    f0.write('\r\n'+'train_mri7, BN_1,2,3, max_pool,batch size=6, cross_entropy,weight_decay=0.001,learning rate=0.00001')
    f0.close()
    data=get_file()
    sum = len(data)
    #sum=324,324=66*4+60
    for i in range(5):
        test_data=[]
        train_data=[]
        begin = 66*i
        end = 66*(i+1)
        if end>sum:
            end = sum
        test_data= data[begin:end]
        train_data = data[0:begin] + data[end:sum]
        run_training(fold=i,train_filenames=train_data,test_filenames=test_data)


if __name__ == '__main__':
    tf.app.run()
