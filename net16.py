# Copyright(c) 2017 Yuhao Tang, Cheng Ouyang, Haoyu Yu, Hong Moon All Rights Reserved.
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# ================================================================================================================

import tensorflow as tf
import numpy as np
import numpy.matlib
#from scipy.misc import imread, imresize
#from imagenet_classes import class_names

class net:   
    def __init__(self, batch_size):
        self.imgs = tf.placeholder(tf.float32, [batch_size, 500, 500, 3])
        self.label = tf.placeholder(tf.float32, [batch_size, 500, 500])
        self.batch_size = batch_size
        self.convlayers()
        self.deconvlayers()
        self.pixel_wise_cross_entropy()
        self.bilinear_kernel(4)

    def bilinear_kernel(self,size,num_class=21):
        """
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
         
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        kernel2d = np.float32((1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor))
        return tf.tile(tf.reshape(kernel2d,[size,size,1,1]),[1,1,num_class,num_class])

    def convlayers(self):
        self.parameters = []
        self.weight = []
        self.bias = []
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.tile(tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean'), [self.batch_size,500,500,1])
            self.images = self.imgs-mean
        
        #-----------------------------------------------------------------------------------------------------------------------
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')
        # -------------------------------------------------------------------------------------------------------------------------

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # -------------------------------------------------------------------------------------------------------------------------

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # -------------------------------------------------------------------------------------------------------------------------

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # -------------------------------------------------------------------------------------------------------------------------

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
        # -------------------------------------------------------------------------------------------------------------------------

        # convolutionalized fc layer
        # conv6 10x10x4096
        with tf.name_scope('conv6') as scope:
            kenerl = kernel = tf.Variable(tf.truncated_normal([7, 7, 512, 4096], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool5, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv6 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # -------------------------------------------------------------------------------------------------------------------------

        # conv7
        with tf.name_scope('conv7') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, 4096], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv6, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv7 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

    def deconvlayers(self):
        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.conv7_drop = tf.nn.dropout(self.conv7, self.keep_prob)
        # 1x1 convolutional 10x10x21
        with tf.name_scope('conv8') as scope:
            #kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, 21], dtype=tf.float32,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.Variable(tf.zeros([1, 1, 4096, 21], dtype=tf.float32), name='weights')
            conv = tf.nn.conv2d(self.conv7_drop, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[21], dtype=tf.float32), name='biases')
            #biases = tf.Variable(tf.truncated_normal([21], dtype=tf.float32), name='biases')
            out = tf.nn.bias_add(conv, biases)
            #self.conv8 = tf.nn.relu(out)
            self.conv8 = out
            self.weight += [kernel]
            self.bias += [biases]
        
        # 1x1 convolutional for pool4
        with tf.name_scope('fc_pool4') as scope:
            kernel = tf.Variable(tf.truncated_normal([1,1,512,21],dtype=tf.float32,stddev=1e-1),name='weights',trainable=True)
            conv = tf.nn.conv2d(self.pool4,kernel,[1,1,1,1],padding='SAME')
            biases = tf.Variable(tf.constant(0.0,shape=[21],dtype=tf.float32),trainable=True,name='biases')
            out = tf.nn.bias_add(conv,biases)
            self.pool4_fc = out
            self.weight += [kernel]
            self.bias += [biases]
            print("the shape of pool4_fc is", self.pool4_fc.get_shape().as_list())

        # bilinear interpolation for conv8
        with tf.name_scope('fuse_fc_pool4_conv8') as scope:
            kernel = tf.Variable(self.bilinear_kernel(4),dtype=tf.float32,name='weights')
            #print(kernel.eval())
            # not very sure if that work
            # pool4 32*32, assume conv8 is 16x16, it is not a perfect 2x expansion
            deconv = tf.nn.conv2d_transpose(self.conv8,kernel,[self.batch_size,32,32,21],[1,2,2,1],padding='SAME')
            biases = tf.Variable(tf.constant(0.0,shape=[21],dtype=tf.float32),
                                                    trainable=True,name='biases')
            out = tf.nn.bias_add(deconv,biases)
            out = tf.add(out,self.pool4_fc,name='fuse_pool4_fc_conv8')
            self.weight += [kernel]
            self.bias += [biases]
            self.fuse_fc_pool4_conv8 = out
            print("the shape of fuse_fc_pool4_conv8 is" ,self.fuse_fc_pool4_conv8.get_shape().as_list())                                               
            

        # deconvolutional layer
        with tf.name_scope('deconv1') as scope:
            self.deconv1 = tf.image.resize_bilinear(self.fuse_fc_pool4_conv8,[500,500])
        #    kernel = tf.Variable(tf.truncated_normal([64, 64, 21, 21],
        #                                                   dtype=tf.float32,
        #                                                  stddev=1e-1), name='weights')
        #    #kernel = tf.zeros([128, 128, 21, 21], dtype=tf.float32, name='weights')
        #    deconv = tf.nn.conv2d_transpose(self.conv8, kernel, [self.batch_size, 5, [1, 32, 32, 1], padding='SAME')
        #    biases = tf.Variable(tf.constant(0.0, shape=[21], dtype=tf.float32),
        #    #biases = tf.Variable(tf.truncated_normal([21], dtype=tf.float32),
        #                                     trainable=True, name='biases')
        #    out = tf.nn.bias_add(deconv, biases)
	#
        #    #  self.deconv1 = tf.nn.relu(out, name=scope)
        #    self.deconv1 = out
        #    self.weight += [kernel]
        #    self.bias += [biases]

        # now add the conv for result of pool4
       
    

    def pixel_wise_cross_entropy(self):
        # self.cross_entropy = tf.reduce_mean(          
        #   tf.nn.softmax_cross_entropy_with_logits(
        #        labels=tf.reshape(self.label, [-1, 21]), logits=tf.reshape(self.deconv1, [-1,21])))
       	self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=tf.to_int32(self.label), logits=self.deconv1)) 
        # weight decay
        lambda_ = 5**(-4)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.cross_entropy += lambda_*l2_loss

    def load_weights(self, weight_file,sess, trainable_weight=None):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i < len(self.parameters):
                sess.run(self.parameters[i].assign(tf.reshape(weights[k], self.parameters[i].get_shape())))
        if trainable_weight is not None:
            train_weight = np.load(trainable_weight)
            sess.run(self.weight[0].assign(train_weight['arr_0']))
            sess.run(self.bias[0].assign(train_weight['arr_1']))
            sess.run(self.weight[1].assign(train_weight['arr_2']))
            sess.run(self.bias[1].assign(train_weight['arr_3']))
            sess.run(self.weight[2].assign(train_weight['arr_4']))
            sess.run(self.bias[2].assign(train_weight['arr_5']))



'''
if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print class_names[p], prob[p]
'''
