'''
Heavily inspired from :


A Dynamic Recurrent Neural Network (LSTM) implementation example using
TensorFlow library. This example is using a toy dataset to classify linear
sequences. The generated sequences have variable length.
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import random

dimMot = 128
dimImgs = 4096
nbExTot = 1000
longMaxLeg = 39
#raj mot deb et mot fin dans les legendes

# ====================
#  DATA READING AND PADDING
# ====================
def litEtTraiteTsLesExemples():
	#identifiantEx = 0
	print("test ")
	#one line = idImage#nbWordsIncludingBEGINandEND#dimensionImgs numbers separated by ' '#diensionWords numbers separated by ' '#idem for next word#...
	#return tf.constant(tab) #containing all exemple, tab must be a numpy table (or a list)
	#the file containing the data must not be padded (to padd : make all legends the exact same length by adding [0 0 0 0 ...] to the shortest ones)
	monFichier = open("trainingSet10000.txt", "r")
	resLeg = []
	resImgs = []
	tailleExs = []
	
	for curL in monFichier :
		if(len(curL) >= 3):
			premSplit = curL.split('#')
			res1Im = []
			estPrem = True
			estDeux = False
			estTer = False
			for curV in premSplit:
				if estDeux :
					split2 = curV.split(' ')
					tab = []
					for val in split2:
						if(len(val) > 0):
							tab += [float(val)]
					resImgs += [[tab]]
					#resImgs += [[[float(val) for val in split2]]]
					estDeux = False
					estTer = True
					#print("Deux", len(tab))
					
				elif estPrem :
					#print("Prem", curV)
					estPrem = False
					estDeux = True
				elif estTer :
					#print("ter", curV)
					tailleExs += [int(curV)]
					estTer = False
				else :
					split2 = curV.split(' ')
					tab = []
					for val in split2:
						if(len(val) > 0):
							tab += [float(val)]
					#print(" ", len(tab))
					res1Im += [tab]
					#print("nbMots ", len(res1Im))
					#res1Im += [[float(val) for val in split2]]
			curLen = len(res1Im)
			res1Im += [[0 for i in range(dimMot)] for i in range(longMaxLeg - curLen)]
			#print("nbMots ", len(res1Im))
			resLeg += [res1Im]
	
	print(len(resLeg))
	"""for i in range(len(resLeg)):
		for j in range(len(resLeg[i])):
			if(len(resLeg[i][j]) != dimMot):
				print(i, " ", j, " ", len(resLeg[i][j]))"""
	#print("fine")
	#print(resLeg[0][0])
	monFichier.close()
	return tailleExs, resLeg, resImgs

class ToySequenceData(object):
    def __init__(self, n_samples=nbExTot, max_seq_len=longMaxLeg, min_seq_len=1):
		self.data = []
		self.labels = []
		self.seqlen = []
		tailleExs, resLeg, resImgs = litEtTraiteTsLesExemples()
		self.data = resLeg
		self.seqlen = tailleExs
		self.labels = [resLeg[idImage][:] for idImage in range(len(resLeg))]
		self.batch_id = 0
		self.dataImgs = resImgs

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
		all batches are of the same size : batch_size (hence if nbExsTot % sizeBatch != 0, we will never see the last exemples)
        """
        if self.batch_id == len(self.data) or len(self.data) < self.batch_id + batch_size:
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_imgs = (self.dataImgs[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen, batch_imgs


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.001
training_iters = 1000
batch_size = 500 #128 avt
display_step = 1

# Network Parameters
seq_max_len = longMaxLeg # Sequence max length ie max number of words in a caption
n_hidden = 2024# hidden layer number of features
n_classes = dimMot

trainset = ToySequenceData(n_samples=nbExTot, max_seq_len=seq_max_len)

with tf.device('/gpu:1'):
	# tf Graph input, contains only yhe captions
	x = tf.placeholder("float", [None, seq_max_len, dimMot])
	y = tf.placeholder("float", [None, seq_max_len, n_classes]) #not +1 or -1 in the dimensions because : for each input, an output but : we don't care about the last output (the one generated by inputing the word END) and we add one more input to the caption words : the image
	# A placeholder for indicating each sequence length
	seqlen = tf.placeholder(tf.int32, [None])
	#the images
	imgs = tf.placeholder("float", [None, 1, dimImgs])

	# Define weights
	weights = {
		'out': tf.Variable(tf.random_normal([n_hidden, n_classes])),
		'in' : tf.Variable(tf.random_normal([dimImgs, n_classes]))
	}
	biases = {
		'out': tf.Variable(tf.random_normal([n_classes])),
		'in': tf.Variable(tf.random_normal([n_classes]))
	}


	def dynamicRNN(x, imgs, seqlen, weights, biases):
		#we get images to the same dimensions as words using a matrix that will be "learned" during training
		imgsBonFormat = [tf.matmul(tf.gather(imgs, i), weights['in']) + biases['in'] for i in range(batch_size)]
		bonX = tf.concat([imgsBonFormat, x], 1)#on concatene mots et image pour chaque exemple
		
		# Prepare data shape to match `rnn` function requirements
		# Current data input shape: (batch_size, n_steps, n_input)
		# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
		# Permuting batch_size and n_steps
		bonX = tf.transpose(bonX, [1, 0, 2])
		# Reshaping to (n_steps*batch_size, n_input)
		bonX = tf.reshape(bonX, [-1, dimMot])
		# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
		bonX = tf.split(num_or_size_splits=seq_max_len + 1, axis = 0, value = bonX)

		# Define a lstm cell with tensorflow
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

		outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, bonX, dtype=tf.float32)
		
		# 'outputs' is a list of output at every timestep 
		#we multiply by W and add Bias a t every timestep
		newOutput = [tf.matmul(curOut, weights['out']) + biases['out'] for curOut in outputs]
		#we rmove the last output (generated after receiving the word END)
		newOutput = tf.stack(newOutput[:-1])
		# and change back dimension to [batch_size, n_step, n_input]
		newOutput = tf.transpose(newOutput, [1, 0, 2])
		return newOutput # dimension : nbExs * (nbSteps - 1) * wordDimension

	pred = dynamicRNN(x, imgs, seqlen, weights, biases)

	# Define loss and optimizer
	mask = tf.sequence_mask(seqlen, seq_max_len)#because seqMaxLen = maxNumerOfInterestingValues = nbWords
	maskedPred = tf.boolean_mask(pred, mask)#for each example, we only keep the values that are not padding (in hat is predicted -> pred, and in the groudTruth)
	maskedTrue = tf.boolean_mask(y, mask)
	cost = tf.reduce_mean(tf.square(tf.subtract(maskedPred, maskedTrue)))#euclidean distance
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

	# Evaluate model
	accuracy = tf.reduce_mean(tf.square(tf.subtract(maskedPred, maskedTrue)))#should be called cost :)

	# Initializing the variables
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

configu = tf.ConfigProto(allow_soft_placement = True)
# Launch the graph
with tf.Session(config = configu) as sess:
    #if training and saving learned weights :
    sess.run(init)
    
    #if using saved weights :
    #saver.restore(sess, "/users/eleves-b/2015/anouk.paradis/model/rapTr25000DimMot20nH2024.ckpt")
    #print("Model restored.")

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen, batch_imgs = trainset.next(batch_size)

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen, imgs: batch_imgs})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen, imgs: batch_imgs})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen, imgs: batch_imgs})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    #if training and saving weights :
    save_path = saver.save(sess, "/users/eleves-b/2015/anouk.paradis/model/rapTr25000DimMot20nH2024bis.ckpt")
    print("Model saved in file: %s" % save_path)
