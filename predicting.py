'''
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
nbExTot = 20
longMaxLeg = 39
#don't forget to add the BEGIN and END word to all captions in the input data

# ====================
#  TOY DATA GENERATOR
# ====================
def litEtTraiteTsLesExemples():
	#one line = idImage#4096 nbs representing the image
	#return tf.constant(tab)
        monFichier = open("imATesterVal20-40Imgs.txt", "r")
	resImgs = []
	idImgs = []
	
	for curL in monFichier :
		premSplit = curL.split('#')
		if len(premSplit) >= 2:
			res1Im = []
			estPrem = True
			for curV in premSplit:
				if estPrem :
					#print(curV)
					idImgs += [curV]
					estPrem = False
				else :
					#split2 = curV.split(' ')
					#resImgs += [[float(val) for val in split2]]
					
					split2 = curV.split(' ')
					tab = []
					for val in split2:
						if(len(val) > 0 and val != "\n"):
							#print(val)
							tab += [float(val)]
					resImgs += [[tab]]
	monFichier.close()
	return idImgs, resImgs

def litDico():
	#one line: a word#its coordinates
	monFichier = open("dico_lemma.txt", "r")
	listeMots = []
	cooMots = []
	j = 0
	
	for curL in monFichier :
		j+=1
		premSplit = curL.split('#')
		estPrem = True
		estBon = True
		estDeb = False
		for curV in premSplit:
			if estPrem :
				if(len(curV) == 0):
					print(j)
					estBon = False
				else:
					listeMots += [curV]
				if curV == "DEB":
					estDeb = True
				estPrem = False
			else :
				if estBon :
					split2 = curV.split(' ')
					tab = []
					for val in split2:
						if(len(val) > 0 and val != "\n"):
							#print(val)
							tab += [float(val)]
					cooMots += [tab]
					if estDeb :
						estDeb = False
						cooDEB = tab[:]
						#print("done", cooDEB)
				
					#split2 = curV.split(' ')
					#cooMots += [[float(val) for val in split2]]
	monFichier.close()
	return listeMots, cooMots, cooDEB
	
idImgs, imgsLues = litEtTraiteTsLesExemples()
listeMots, cooMots, cooDEB = litDico()
	
def dist(coo1, coo2):
	res = 0
	for i in range(len(coo1)):
		res += (coo1[i] - coo2[i]) * (coo1[i] - coo2[i])
	return res

def motPlusProche(listeCoo):
	motsPP = []
	for i in range(len(listeCoo)):
		motsPP += [(0, dist(listeCoo[i], cooMots[0]))]
	for i in range(len(cooMots)):
		for motGen in range(len(motsPP)):
			(oldId, oldDist) = motsPP[motGen]
			if dist(listeCoo[motGen], cooMots[i]) < oldDist:
				motsPP[motGen] = (i, dist(listeCoo[motGen], cooMots[i]))
	resu = []
	for (i, cooM) in motsPP:
		resu += [cooMots[i]]
	return resu

# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.001
training_iters = 10000#1000000
batch_size = 20 #128 avt
display_step = 10

# Network Parameters
seq_max_len = longMaxLeg # Sequence max length ie nbMotsMax dans la legende
n_hidden = 2024 # hidden layer num of features
n_classes = dimMot

with tf.device('/gpu:0'):    
	x = tf.placeholder("float", [None, seq_max_len, dimMot])
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
		#get images into the right dimension using the learned matrix
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
		#we multiply by W and add bias at every step
		newOutput = [tf.matmul(curOut, weights['out']) + biases['out'] for curOut in outputs]
		#we take out the last word
		newOutput = tf.stack(newOutput[:-1])
		# and change back dimension to [batch_size, n_step, n_input]
		newOutput = tf.transpose(newOutput, [1, 0, 2])
		return newOutput # dimension : nbExs * nbSteps - 1 * wordDimension

	output = dynamicRNN(x, imgs, seqlen, weights, biases)

	# Initializing the variables
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
configu = tf.ConfigProto(allow_soft_placement = True, gpu_options=gpu_options)

# Launch the graph
#with tf.Session() as sess:
with tf.Session(config = configu) as sess:
    sess.run(init)
    
    saver.restore(sess, "/users/eleves-b/2015/anouk.paradis/model/rapTr25000nH2024Ite1000000.ckpt")
    print("Model restored.")

    legendes = [[cooDEB] + [[0.0 for j in range(dimMot)] for i in range(longMaxLeg - 1)] for i in range(len(idImgs))]
    seqLen = [longMaxLeg for i in range(len(idImgs))]
    print("legendes :", len(legendes))
    print(len(legendes[0]), len(legendes[0][0]))
    print("imgs :", len(imgsLues))
    
    #print("idImgs", idImgs)
    #print("legendes", legendes)
    #print("seqlen", seqLen)
    
    for i in range(longMaxLeg - 1):
		print("mot no", i)
		res = sess.run(output, {x: legendes, seqlen: seqLen, imgs: imgsLues})
		nouvMots = []
		for curI in range(len(legendes)):
			nouvMots += [res[curI][i + 1]]
		bonMots = motPlusProche(nouvMots)
		print(bonMots)
		for curI in range(len(legendes)):
			legendes[curI][i + 1] = bonMots[curI]
		
    #res = sess.run({"out": output, "state":states}, {x: legendes, seqlen: seqLen, imgs: imgsLues})
		#print("output ", res)
	
    fichSortie = open("res20-40ImgsVal.txt","w")
    #print(legendes)
    for i in range(len(idImgs)):
		fichSortie.write(idImgs[i])
		fichSortie.write("#")
		for j in range(longMaxLeg):
			for k in range(dimMot):
				fichSortie.write(str(legendes[i][j][k]))
				fichSortie.write(' ')
			fichSortie.write("#")
		fichSortie.write("\n")
