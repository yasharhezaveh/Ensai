
from PIL import Image
import tensorflow as tf
import scipy.ndimage
from scipy import misc
from scipy.interpolate import RectBivariateSpline
import numpy as np
import numpy.matlib as ml
import random
import time
import os
import gc
import scipy.io
slim = tf.contrib.slim

print("starting the job...")


num_out = 5    #number ouf output parameters being predicted

global numpix_side 
numpix_side = 192   #number of image pixels on the side

global max_noise_rms, max_psf_rms , max_cr_intensity
max_trainoise_rms = 0.1 # maximum rms of noise in training data
max_testnoise_rms = 0.1 # maximum rms of noise in test or validation data
max_noise_rms = max_testnoise_rms

max_psf_rms = 0.08/0.04  # maximum Gaussian PSF rms (in pixels)
max_cr_intensity = 0.5 # maximum scaling for cosmic ray and artefact maps

global constant_noise_rms
variable_noise_rms = True  #if True, the noise rms will be chosen randomly for each sample with a max of max_noise_rms (above)


cycle_batch_size = 50   # how many examples to read at a time (here it's equal to the batch size)
num_test_samples = 1000 # number of test samples

global pix_res
pix_res = 0.04 # pixel size in arcsec
L_side = pix_res * numpix_side

global arcs_data_path_1, arcs_data_path_2 , test_data_path_1 , test_data_path_2 , CRay_data_path
global lens_data_path_1, lens_data_path_2, testlens_data_path_1, testlens_data_path_2 

global min_unmasked_flux 
min_unmasked_flux = 0.75



#number of folders containing training or test data. If all 3 point to the same folder that's OK (only that folder will be used).
global num_data_dirs
num_data_dirs = 3

num_training_samples = 100000
max_num_test_samples = 1000
arcs_data_path_1 = 'data/SAURON_TEST/'
arcs_data_path_2 = 'data/SAURON_TEST/'
arcs_data_path_3 = 'data/SAURON_TEST/'
test_data_path_1 = 'data/SAURON_TEST/'
test_data_path_2 = 'data/SAURON_TEST/'
test_data_path_3 = 'data/SAURON_TEST/'

lens_data_path_1 = 'data/SAURON_TEST/'
lens_data_path_2 = 'data/SAURON_TEST/'
lens_data_path_3 = 'data/SAURON_TEST/'
testlens_data_path_1 = 'data/SAURON_TEST/'
testlens_data_path_2 = 'data/SAURON_TEST/'
testlens_data_path_3 = 'data/SAURON_TEST/'

#folder containing cosmic rays
CRay_data_path  = 'data/CosmicRays/'

global max_xy_range   # xy range of center of the lens. The image is shifted in a central area with a side of max_xy_range (arcsec) during training or testing
max_xy_range = 0.5

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

execfile("get_data.py")

############################################




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MODEL DEFINITION

x = tf.placeholder(tf.float32, shape=[None, numpix_side*numpix_side])   #placeholder for input image
y_ = tf.placeholder(tf.float32, shape=[None,num_out])    #placeholder for output parameters during training
x_image0 = tf.reshape(x, [-1,numpix_side,numpix_side,1])



# removing image intensity bias: filter image with a 4X4 filter and remove from image
MASK = tf.abs(tf.sign(x_image0))
XX =  x_image0 +  ( (1-MASK) * 1000.0)
bias_measure_filt = tf.constant((1.0/16.0), shape=[4, 4, 1, 1])
bias_measure = tf.nn.conv2d( XX , bias_measure_filt , strides=[1, 1, 1, 1], padding='VALID')
im_bias = tf.reshape( tf.reduce_min(bias_measure,axis=[1,2,3]) , [-1,1,1,1] )
x_image = x_image0 - (im_bias * MASK )




# construct models
execfile("ensai_model.py")


y_mod = []
with tf.variable_scope("ENSAI"):
	y_mod.insert( 0 , model_1(x_image,scope="EN_Model1"  ))
	y_mod.insert( 1 , model_2(x_image,scope="EN_Model2"  ))
        y_mod.insert( 2 , model_3(x_image,scope="EN_Model3"  ))
        y_mod.insert( 3 , model_4(x_image,scope="EN_Model4"  ))
        y_mod.insert( 4 , model_5(x_image,scope="EN_Model5"  ))
        y_mod.insert( 5 , model_6(x_image,scope="EN_Model6"  ))
        y_mod.insert( 6 , model_7(x_image,scope="EN_Model7"  ))
        y_mod.insert( 7 , model_8(x_image,scope="EN_Model8"  ))
        y_mod.insert( 8 , model_9(x_image,scope="EN_Model9"  ))
        y_mod.insert( 9 , model_10(x_image,scope="EN_Model10"  ))
	execfile("nets/inception_utils.py")
	execfile("nets/inception.py")
	arg_scope = inception_v4_arg_scope()
	input_tensor =  tf.reshape(x, [-1,numpix_side,numpix_side,1])
	input_tensor = tf.concat([input_tensor,input_tensor,input_tensor], axis=3)
	with tf.variable_scope("EN_Model11"):
		with slim.arg_scope(arg_scope):
			y_mod.insert(10,  inception_v4( input_tensor , num_classes = 5 , dropout_keep_prob=1.0 , is_training=False,create_aux_logits=False) )


#chose model: 5:OverFeat, 8:our made-up model, 9:AlexNet, 11:Inception.v4. Defaults to AlexNet if nothing specified 
if 'model_num' not in locals():
	print "No model selected. Selecting default model (9: AlexNet)."
	model_num = 9
y_conv = y_mod[model_num-1] 



variables_to_save =  slim.get_variables(scope="ENSAI/EN_Model" + str(model_num) )   #list of variables to save
variables_to_restore = variables_to_save   #list of variables to restore (same as save here)
train_pars = variables_to_save  #list of parameters to train



save_file =  "data/trained_weights/model_" + str(model_num) + ".ckpt"     #path of file to save
restore_file = save_file   #path of network weights file to restore from

RESTORE = True
SAVE = False
restorer = tf.train.Saver(variables_to_restore)
saver = tf.train.Saver(variables_to_save)



############## flipping and cost function
MeanSquareCost , y_conv_flipped = cost_tensor(y_conv)




