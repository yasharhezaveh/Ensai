
# testing each model individually

model_to_fit = 9  # pick a model:    5:OverFeat, 8:our made-up model, 9:AlexNet, 11:Inception.v4.
y_conv = y_mod[model_to_fit-1]
MeanSquareCost , y_conv_flipped = cost_tensor(y_conv)

variables_to_restore =  slim.get_variables(scope="ENSAI/EN_Model" + str(model_to_fit) )    #list of variables to restore
restore_file = "data/trained_weights/model_" + str(model_to_fit) + ".ckpt"    #path of file with network weights
restorer = tf.train.Saver([v for v in variables_to_restore if "Adam" not in v.name])    # a tf.train.Saver object used for restoring (or saving)




execfile("get_data.py")    #read the files for generating data



max_xy_range = 0.5  # xy range of center of the lens. The image is shifted in a central area with a side of max_xy_range (arcsec) during training or testing
variable_noise_rms = True   #if True, the noise rms will be chosen randomly for each sample with a max of max_noise_rms
max_noise_rms = 0.1  # maximum rms of noise data
num_samp = 1000   #number of test samples
chunk_size = 50    # batch number: how many test examples to pass at one time.

X = np.zeros( ( num_samp , numpix_side * numpix_side ), dtype='float32') ;   #numpy array holding the images
Y = np.zeros( ( num_samp , num_out ) , dtype='float32' );                    #numpy array holding the lens parameters (here only used to flip for the x-y ellipticity)
Predictions = np.zeros( ( num_samp , num_out ) , dtype='float32' );          #predicted parameters
mag = np.zeros((num_samp,1))
read_data_batch( X , Y , mag , max_num_test_samples  , 'test')             #read data




sess = tf.Session()         #launch a tf session
sess.run(tf.global_variables_initializer())      #initialize variables
restorer.restore(sess, restore_file)             # restore our saved weights


cost = 0.0
ind_t = range(num_samp)
sum_rms = 0
num_chunks = num_samp/chunk_size

#loop over our samples (since we can't give all the test data at once because of limited gpu memory)
for it in range(num_chunks):
	print it
	xA = X[ind_t[0+chunk_size*it:chunk_size+chunk_size*it]]
	yA = Y[ind_t[0+chunk_size*it:chunk_size+chunk_size*it]]	
	cost  = cost + sess.run(MeanSquareCost, feed_dict={x: xA, y_: yA})   # evaluate cost
	A = sess.run(y_conv , feed_dict={ x: xA})   # A is the network prediction for parameters
	B = sess.run(y_conv_flipped , feed_dict={ x: xA})  # B is the same prediction with the ellipticity flipped
	Predictions[ind_t[0+chunk_size*it:chunk_size+chunk_size*it],:]  = get_rotation_corrected(A,B,Y[ind_t[0+chunk_size*it:chunk_size+chunk_size*it],:])  # "Prediction" is now corrected for the flip.
	sum_rms = sum_rms + np.std(Predictions[ind_t[0+chunk_size*it:chunk_size+chunk_size*it],:] -Y[ind_t[0+chunk_size*it:chunk_size+chunk_size*it],:],axis=0)
	print(np.array_str( sum_rms/(it+1)  ,precision=2) )








