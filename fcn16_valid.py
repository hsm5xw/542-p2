# Copyright(c) 2017 Yuhao Tang, Cheng Ouyang, Haoyu Yu, Hong Moon All Rights Reserved.
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# ================================================================================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import net16_val
import time

fhand = open('val_input')
fn = fhand.read().strip()
fhand.close()
val_input_list =fn.split('\n')

fhand = open('val_label')
fn = fhand.read().strip()
fhand.close()
val_label_list =fn.split('\n')

WIDTH = 500
HEIGHT = 500
NUM_EPOCHS = 0
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
	min_queue_examples = 128
	num_threads = 4
	input_ = single_JPEGimage_reader(input_queue)
	label = single_PNGimage_reader(label_queue)
	input_.set_shape([HEIGHT,WIDTH,3])
	label.set_shape([HEIGHT,WIDTH,1])
	input_batch, label_batch = tf.train.shuffle_batch(
		[input_, label],
		batch_size=BATCH_SIZE,
		num_threads=num_threads,
		capacity=min_queue_examples*2, # + (num_threads+2)*BATCH_SIZE,
		seed=0,
		min_after_dequeue=min_queue_examples)
	return input_batch, label_batch

def next_batch(input_queue, label_queue):
	input_batch, label_batch = train_label_reader(input_queue, label_queue)
	batch = tf.concat([input_batch, label_batch], axis=3)
	return batch

def p_acc(confusion_matrix):
	s_nii = 0
	s_ti = 0
	for i in range(21):
		s_nii += confusion_matrix[i,i]
		s_ti += np.sum(confusion_matrix[i,:])
	return s_nii/s_ti

def miou(confusion_matrix):
	iou = 0
	ncl = 0
	for i in range(21):
		sum_nji = np.sum(confusion_matrix[:,i])
		ti = np.sum(confusion_matrix[i,:])
		nii = confusion_matrix[i,i]
		denom = ti + sum_nji - nii
		if (denom != 0):
			iou += nii/denom
			ncl += 1
	return iou/21

def mean_acc(confusion_matrix):
	ncl = 0
	acc = 0
	for i in range(21):
		nii = confusion_matrix[i,i]
		ti = np.sum(confusion_matrix[i,:])
		if (ti != 0):
			acc += nii/ti
			ncl += 1
	return acc/21

def fw_IU(confusion_matrix):
	iou = 0
	tcl = 0
	for i in range(21):
		sum_nji = np.sum(confusion_matrix[:,i])
		ti = np.sum(confusion_matrix[i,:])
		nii = confusion_matrix[i,i]
		denom = ti + sum_nji - nii
		if (denom != 0):
			iou += ti*nii/denom
			tcl += ti
	return iou/tcl

sess = tf.InteractiveSession()
# CNN 
fcn_vgg16 = net16_val.net(BATCH_SIZE)
print('CNN constructed...')

# input pipeline
# validation
val_input_queue = tf.train.string_input_producer(val_input_list, shuffle=False)
val_label_queue = tf.train.string_input_producer(val_label_list, shuffle=False)
valide_batch = next_batch(val_input_queue, val_label_queue)
print('pipe constructed...')

# initialize
init_glb = tf.global_variables_initializer()
init_loc = tf.local_variables_initializer()
sess.run(init_glb)
sess.run(init_loc)
fcn_vgg16.load_weights(sess, weight_file=None, trainable_weight='weight.npz', update_weight='param_vgg_update.npz')
coord = tf.train.Coordinator()
print('start ...')
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Validation
l = len(val_label_list)
# p_acc = 0 # pixel accuracy
# m_acc = 0
# m_iou = 0
# fw_iou = 0
cm_ = np.zeros((21,21))
for i in range(l//BATCH_SIZE):
	print("the %s th of %s sample" %(i,l//BATCH_SIZE))
	start_time = time.time()

	tmp = valide_batch.eval()
	val_input_ = tmp[:,:,:,:3]
	val_label_ = tmp[:,:,:,3]
	# p_acc += fcn_vgg16.accuracy.eval(feed_dict={fcn_vgg16.imgs:val_input_, fcn_vgg16.label:val_label_, fcn_vgg16.keep_prob:1})
	# print(p_acc)
	cm_ += fcn_vgg16.cm.eval(feed_dict={fcn_vgg16.imgs:val_input_, fcn_vgg16.label:val_label_, fcn_vgg16.keep_prob:1})
	# m_acc += mean_acc(cm_)
	# m_iou += miou(cm_)
	# fw_iou += fw_IU(cm_)
	# print(m_acc)
	# print(m_iou)
	# print(fw_iou)
	#fwm_IU += fcn_vgg16.fwMeanIU.eval(feed_dict={fcn_vgg16.imgs:val_input_, fcn_vgg16.label:val_label_, fcn_vgg16.keep_prob:1})
	print('Validation: time elapsed: %.3fs.'%(time.time()-start_time))

print("pixel accuracy: ",p_acc(cm_))
print("mean pixel accuracy: ",mean_acc(cm_))
print("mean IoU: ", miou(cm_))
print("frequency weighted IU: ",fw_IU(cm_))
coord.request_stop()











