reuse = False
weight_prefix = "model_"

FLIPXY = tf.constant([ [1., 0., 0., 0., 0.],[0. , -1. , 0., 0., 0.],[0., 0. , -1., 0., 0.],[0., 0. , 0., 1., 0.],[0., 0. , 0., 0., 1.]] )

models_to_combine = [5,8,9,11]
num_models = len(models_to_combine)

with tf.variable_scope("ENSAI",reuse = reuse):
        combi_weights = tf.get_variable("combi_weights", [num_out*num_models,num_out] , initializer=tf.constant_initializer(1.0/np.double(num_models) ) )
Combi_to_save = slim.get_variables(scope="ENSAI/combi_weights"  )

variables_to_restore = []
restore_file = []
restorer = []
n = 0
for i_mod in models_to_combine:
	variables_to_restore.append( slim.get_variables(scope="ENSAI/EN_Model" + str(i_mod) ) )
	restore_file.append(  "data/trained_weights/model_" + str(i_mod) + ".ckpt" )
	restorer.append( tf.train.Saver([v for v in variables_to_restore[n] if "Adam" not in v.name]) )
	n = n + 1




DIAG1 = tf.diag([ 1. , 1. , 1. , 1. , 1.]) 
enforce_banded = tf.concat(axis=0, values=[DIAG1] * num_models )
banded_combi = combi_weights * enforce_banded



#the issue of combining different networks for ellipticity xy is again tricky: We compute all different combinations of ex,ey ==> -ex,-ey for the 4 predictions and average them to minimize the inter-model cost
y_mod_flipped = [y_mod[models_to_combine[0]-1]]
for i_mod in models_to_combine[1:]:
	elpcost1 = tf.reshape( tf.reduce_mean( tf.pow( y_mod[models_to_combine[0]-1][:,1:3] - y_mod[i_mod-1][:,1:3] , 2)   ,axis=1) , [-1,1] )
	elpcost2 = tf.reshape( tf.reduce_mean( tf.pow( y_mod[models_to_combine[0]-1][:,1:3] + y_mod[i_mod-1][:,1:3] , 2)   ,axis=1) , [-1,1] )
	elp_ind  = tf.reshape( tf.argmax( tf.concat ( axis=1 , values=[elpcost1, elpcost2] ) , axis=1 )  , [-1,1] )
	flip_or_not = tf.to_float(elp_ind*2-1)
	a0 = tf.reshape(tf.sign(tf.abs(y_mod[i_mod-1][:,0])) ,[-1,1] )
	flipxy_tensor = tf.concat ( axis=1 , values=[a0, flip_or_not , flip_or_not , a0 , a0 ] )
	y_mod_flipped.append( flipxy_tensor * y_mod[i_mod-1] )


Y_stack = tf.concat(axis=1, values= y_mod_flipped )
y_conv =  tf.matmul( Y_stack , banded_combi)


MeanSquareCost , y_conv_flipped = cost_tensor(y_conv)

Combi_saver = tf.train.Saver(Combi_to_save)
Combi_file =  "data/trained_weights/Trained_Combi.ckpt"

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i_mod in range(num_models):
	restorer[i_mod].restore(sess, restore_file[i_mod])
	




#num_test_samples = 50
#X_test = np.zeros((num_test_samples,numpix_side*numpix_side), dtype='float32'  );
#Y_test = np.zeros((num_test_samples,num_out), dtype='float32' );
#read_data_batch( X_test , Y_test , num_test_samples , 'test' )






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
        print("rms in the parameter difference: " + np.array_str( sum_rms/(it+1)  ,precision=2) )


