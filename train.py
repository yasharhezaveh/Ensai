

############### OPTIMIZER:
learning_rate = 1e-6
train_step = tf.train.AdamOptimizer(learning_rate).minimize(MeanSquareCost,var_list=train_pars)
##########################


num_batch_samples = 50
num_iterations = 1
min_eval_cost = 0.06
X = np.zeros((cycle_batch_size,numpix_side*numpix_side), dtype='float32') ;
Y = np.zeros((cycle_batch_size,num_out), dtype='float32' );
MAG = np.zeros((cycle_batch_size,1), dtype='float32' );


min_unmasked_flux = 0.98
X_test = np.zeros((num_test_samples,numpix_side*numpix_side), dtype='float32'  );
Y_test = np.zeros((num_test_samples,num_out), dtype='float32' );
MAG_test = np.zeros((num_test_samples,1), dtype='float32' );
max_noise_rms = max_testnoise_rms
read_data_batch( X_test , Y_test , MAG_test , max_num_test_samples , 'test' )




sess = tf.Session()
sess.run(tf.global_variables_initializer())

if RESTORE:
        restorer.restore(sess, restore_file)



n = 0
ind_t = range(num_test_samples)
train_cost = 0
write_time = time.time()
start_time = time.time()

log_file = open("log_file.txt","w")
log_file.close()

for i_sample in range(1000000):
        if i_sample%1 == 0:

		max_noise_rms = max_trainoise_rms
		min_unmasked_flux = 0.75
                read_data_batch( X , Y , MAG , num_training_samples , 'train' )

        for i in range(num_iterations):
                n = n + 1

		if cycle_batch_size==num_batch_samples:
	                ind = range(num_batch_samples)
		else:
			ind = np.random.randint(0, high=cycle_batch_size, size=num_batch_samples)

                xA = X[ind]
                yA = Y[ind]




		# once every 20 iterations evaluate things for the validation set.
                print_per = 20
                if n%print_per == 1:
			gc.collect()
                        train_cost = sess.run(MeanSquareCost, feed_dict={x:xA, y_: yA} )

                        sum_rms = 0
                        eval_cost = 0
                        num_chunks = 20
                        for it in range(num_chunks):
                                eval_cost  = eval_cost + sess.run(MeanSquareCost, feed_dict={x: X_test[ind_t[0+50*it:50+50*it]], y_: Y_test[ind_t[0+50*it:50+50*it],:]})
                                A = sess.run(y_conv , feed_dict={ x: X_test[ind_t[0+50*it:50+50*it]]})
                                B = sess.run(y_conv_flipped , feed_dict={ x: X_test[ind_t[0+50*it:50+50*it]]})
                                ROT_COR_PARS = get_rotation_corrected(A,B,Y_test[ind_t[0+50*it:50+50*it],:])
                                sum_rms = sum_rms + np.std(ROT_COR_PARS-Y_test[ind_t[0+50*it:50+50*it],:],axis=0)
                        eval_cost = eval_cost / num_chunks
                        print("mod "+ str(model_num) + ", lr: " + str(learning_rate) + ", "  + np.array_str( sum_rms/num_chunks  ,precision=2) )


			# show the iteration number, training cost, validation cost, and the average time per iteration for training
                        print("                                         %0.4d    %0.4d    %0.5f    %0.5f    %0.5f   %0.3f"%(i_sample,i,train_cost,eval_cost,min_eval_cost,(time.time()-start_time)/print_per)) 
                        start_time = time.time()
                        
                        log_file = open("log_file.txt","a")
                        log_file.write('%d ' % (i_sample) + ' '.join(map(str,sum_rms/num_chunks)) + ' %0.5f %0.5f\n' % (train_cost,eval_cost) )
                        log_file.close()
                        
                        if  SAVE & (eval_cost<min_eval_cost) & (n>20): # save file when validation cost drops
                                print "saving weights to the disk (eval) ..."
                                save_path = saver.save(sess, save_file)
                                print "done."
                        min_eval_cost = np.minimum(min_eval_cost,eval_cost)
                sess.run(train_step, feed_dict={x: xA, y_: yA})

