# Copyright(c) 2017 Yuhao Tang, Cheng Ouyang, Haoyu Yu, Hong Moon All Rights Reserved.
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# ================================================================================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import net16
import time

fhand = open('input')
fn = fhand.read().strip()
fhand.close()
input_list = fn.split('\n')

fhand = open('label1')
fn = fhand.read().strip()
fhand.close()
label_list =fn.split('\n')

# fhand = open('val_input')
# fn = fhand.read().strip()
# fhand.close()
# val_input_list =fn.split('\n')

# fhand = open('val_label')
# fn = fhand.read().strip()
# fhand.close()
# val_label_list =fn.split('\n')

WIDTH = 500
HEIGHT = 500
NUM_EPOCHS = 1
BATCH_SIZE = 1

def toClass_(array):
	gt = np.zeros((BATCH_SIZE,HEIGHT,WIDTH,21))
	for idx in range(BATCH_SIZE):
		for i in range(HEIGHT):
			for j in range(WIDTH):
				l = int(array[idx,i,j])
				if (l != 255):
					gt[idx,i,j,l] = 1
				else:
					gt[idx,i,j,0] = 1
	return gt

def segment(out_):
	seg = np.argmax(out_, axis=3)
	return seg

def single_JPEGimage_reader(filename_queue):
	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)
	image = (tf.to_float(tf.image.decode_jpeg(image_file, channels=3)))
	image = tf.image.resize_images(image,[HEIGHT,WIDTH],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	return image

def single_PNGimage_reader(filename_queue):
	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)
	image = tf.to_float(tf.image.decode_png(image_file, channels=1))
	image = tf.image.resize_images(image,[HEIGHT,WIDTH],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	# pixel distribution ground truth 
	return image

def train_label_reader(input_queue, label_queue):
	min_queue_examples = 32
	num_threads = 1
	input_ = single_JPEGimage_reader(input_queue)
	label = single_PNGimage_reader(label_queue)
	input_.set_shape([HEIGHT,WIDTH,3])
	label.set_shape([HEIGHT,WIDTH,1])
	input_batch, label_batch = tf.train.shuffle_batch(
		[input_, label],
		batch_size=BATCH_SIZE,
		num_threads=num_threads,
		capacity=min_queue_examples + (num_threads+2)*BATCH_SIZE,
		seed=0,
		min_after_dequeue=min_queue_examples)
	return input_batch, label_batch

def next_batch(input_queue, label_queue):
	input_batch, label_batch = train_label_reader(input_queue, label_queue)
	batch = tf.concat([input_batch, label_batch], axis=3)
	return batch

sess = tf.InteractiveSession()
# CNN 
fcn_vgg16 = net16.net(BATCH_SIZE)
train_step = tf.train.MomentumOptimizer(1e-4,0.9).minimize(fcn_vgg16.cross_entropy)
#print('warning, the original learning rate should be 1e-4'
print('CNN constructed...')

# input pipeline
# training
input_queue = tf.train.string_input_producer(input_list, shuffle=False)
label_queue = tf.train.string_input_producer(label_list, shuffle=False)
train_batch = next_batch(input_queue, label_queue)
# validation
# val_input_queue = tf.train.string_input_producer(val_input_list, shuffle=False)
# val_label_queue = tf.train.string_input_producer(val_label_list, shuffle=False)
print('pipe constructed...')

# initialize
init = tf.global_variables_initializer()
sess.run(init)
fcn_vgg16.load_weights('vgg16_weights.npz',sess,'weight.npz')
coord = tf.train.Coordinator()
print('start ...')
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#train
loss = []
for i in range(NUM_EPOCHS):
	print('epoch %d ...'%i)
	tmp = train_batch.eval()
	input_ = tmp[:,:,:,:3]
	label_ = tmp[:,:,:,3]
	start_time = time.time()
	train_step.run(feed_dict={fcn_vgg16.imgs:input_, fcn_vgg16.label:label_, fcn_vgg16.keep_prob:0.85})
	print('Train: time elapsed: %.3fs.'%(time.time()-start_time))
	if i%100 == 0:
		start_time = time.time()
		loss += [fcn_vgg16.cross_entropy.eval(feed_dict={fcn_vgg16.imgs:input_, fcn_vgg16.label:label_, fcn_vgg16.keep_prob:1})]
		print('Evaluate Loss: time elapsed: %.3fs.'%(time.time()-start_time))
		print('max_w1 weight:')
		print(np.max(fcn_vgg16.weight[0].eval()))
	
start_time = time.time()
seg_dist = fcn_vgg16.deconv1.eval(feed_dict={fcn_vgg16.imgs:input_, fcn_vgg16.label:label_, fcn_vgg16.keep_prob:1})
print('time elapsed: %.3fs.'%(time.time()-start_time))
seg_ = segment(seg_dist)
print('warning: image not rescaled!')
print(loss)
np.savez('segment.npz',seg_)
np.savez('label_.npz',label_)
np.savez('loss.npz',np.array(loss))
np.savez('seg_dict',seg_dist)
w1 = fcn_vgg16.weight[0].eval()
b1 = fcn_vgg16.bias[0].eval()
np.savez('weight.npz',w1,b1)

# Validation
#l = len(val_label_list)
#p_acc = 0 # pixel accuracy
# for i in range(l):
# 	plt.figure()
# 	val_input_ = single_JPEGimage_reader(val_input_queue).eval()
# 	val_label_ = single_PNGimage_reader(val_label_queue).eval()
# 	p_acc += fcn_vgg16.PixelAccuracy.eval(feed_dict={fcn_vgg16.imgs:val_input_, fcn_vgg16.label:val_label_, fcn_vgg16.keep_prob:1})
# 	m_p_acc += fcn_vgg16.MeanPixelAccuracy.eval(feed_dict={fcn_vgg16.imgs:val_input_, fcn_vgg16.label:val_label_, fcn_vgg16.keep_prob:1})
# 	m_IU += fcn_vgg16.MeanIU.eval(feed_dict={fcn_vgg16.imgs:val_input_, fcn_vgg16.label:val_label_, fcn_vgg16.keep_prob:1})
# 	fwm_IU += fcn_vgg16.fwMeanIU.eval(feed_dict={fcn_vgg16.imgs:val_input_, fcn_vgg16.label:val_label_, fcn_vgg16.keep_prob:1})

coord.request_stop()











