Y_all_train=[[],[],[]]
Y_all_test =[[],[],[]]

Y_all_train[0] = np.loadtxt(arcs_data_path_1 + '/parameters_train.txt')
Y_all_test[0] = np.loadtxt(test_data_path_1 + '/parameters_test.txt')

Y_all_train[1] = np.loadtxt(arcs_data_path_2 + '/parameters_train.txt')
Y_all_test[1] = np.loadtxt(test_data_path_2 + '/parameters_test.txt')

Y_all_train[2] = np.loadtxt(arcs_data_path_3 + '/parameters_train.txt')
Y_all_test[2] = np.loadtxt(test_data_path_3 + '/parameters_test.txt')


R_n = np.loadtxt( 'data/PS_4_real.txt')
I_n = np.loadtxt( 'data/PS_4_imag.txt')


xv, yv = np.meshgrid( np.linspace(-L_side/2.0, L_side/2.0, num=numpix_side) ,  np.linspace(-L_side/2.0, L_side/2.0, num=numpix_side))


def get_rotation_corrected( A , B , C ):    # this code flips the ellipticity so that (-ex -ey) and (ex, ey) are both evaluated and the best combination is kept (parity invariance)
    D = np.mean((C-A)**2,axis=1)
    E = np.mean((C-B)**2,axis=1)
    ind1 = np.where(D<=E)
    ind2 = np.where(D>E)
    ROT_COR_PARS = np.zeros((A.shape[0],5))
    ROT_COR_PARS[:,0] = A[:,0]
    ROT_COR_PARS[ind1,1] = A[ind1,1]
    ROT_COR_PARS[ind2,1] = B[ind2,1]
    ROT_COR_PARS[ind1,2] = A[ind1,2]
    ROT_COR_PARS[ind2,2] = B[ind2,2]
    ROT_COR_PARS[:,3] = A[:,3]
    ROT_COR_PARS[:,4] = A[:,4]
    return ROT_COR_PARS

def read_batch_online( X , Y , max_file_num , train_or_test):
	num_samp = X.shape[0]
	Xmat, Ymat = eng.online_image_generator(num_samp , -1 , numpix_side , os.environ['LOCAL_SCRATCH'] , nargout=2)
	Xmat = np.array(Xmat._data.tolist())
	Xmat = Xmat.reshape((num_samp,numpix_side,numpix_side))
	Xmat = np.transpose(Xmat, axes=(0,2,1)).reshape((num_samp,numpix_side*numpix_side))
	Ymat = np.array(Ymat._data.tolist())
	Ymat = Ymat.reshape((num_out,num_samp)).transpose()
	X[:] = Xmat
	Y[:] = Ymat



def make_real_noise(Fmap):
    Npix = Fmap.shape[0];
    Npix_2 = Npix/2;
    Npix_2p1 = Npix/2 + 1;
    Npix_2p2 = Npix/2 + 2;
    Npix_2m1 = Npix/2 - 1;

    np.conj(np.fliplr(np.flipud(Fmap[Npix_2p1-1,1:Npix_2-1].reshape((-1,1))))).shape

    A = np.concatenate( (Fmap[0:Npix_2,Npix_2p2-1:] , np.conj(np.fliplr(np.flipud(Fmap[Npix_2p1-1,1:Npix_2].reshape((1,-1))))) ) , axis = 0)
    B = np.concatenate( (Fmap[0:Npix_2p1,0:Npix_2p1], A) , axis = 1)
    C = np.concatenate( (np.zeros((Npix_2m1,1)) , np.conj(np.fliplr(np.flipud(Fmap[1:Npix_2,Npix_2p2-1:]))), np.conj(np.fliplr(np.flipud(Fmap[1:Npix_2,1:Npix_2p1]))) ) , axis = 1)
    sym_fft = np.concatenate( (B ,C ) , axis = 0)
    noise_map = np.real( np.fft.ifft2(np.fft.ifftshift(sym_fft)) ) 
    noise_map = noise_map/np.std(noise_map)
    return noise_map

def add_gaussian_noise(im):
    if variable_noise_rms == False:
    	rnd_noise_rms=max_noise_rms
    else:
	rnd_noise_rms = np.random.uniform(low=max_noise_rms/10, high=max_noise_rms)

    if np.random.uniform(low=0, high=1)<=0.25:
    	noise_map = np.random.normal(loc=0.0, scale = rnd_noise_rms,size=im.shape)
    else:
	FFT_NOISE = np.random.normal(loc=0.0, scale = np.abs(R_n))  + np.random.normal(loc=0.0, scale = np.abs(I_n) ) *1j
	noise_map = make_real_noise(FFT_NOISE)
    	noise_map = rnd_noise_rms * noise_map
	noise_map = noise_map.reshape((1,-1))
    im[:] = im[:] + noise_map



def gen_masks(nmax,ARCS , apply_prob=0.5):
        mask = 1.0
	if np.min(ARCS)<0.1 and np.max(ARCS)>0.9:
        	if np.random.uniform(low=0, high=1)<=apply_prob:
			while True:
                		mask = np.ones((numpix_side,numpix_side),dtype='float32')
                		num_mask = np.random.randint(1, high = nmax)
                		for j in range(num_mask):
                        		x_mask =  np.random.uniform(low=-L_side/2.0, high=L_side/2.0)
                        		y_mask =  np.random.uniform(low=-L_side/2.0, high=L_side/2.0)
                        		r_mask = np.sqrt( (xv- x_mask  )**2 + (yv- y_mask )**2 )
                        		mask_rad = 0.2
                        		mask = mask * np.float32(r_mask>mask_rad)
				if np.sum(mask*ARCS) >= ( min_unmasked_flux * np.sum(ARCS)):
					break
	return mask


def apply_psf(im , my_max_psf_rms , apply_prob=1.0 ):
        np.random.uniform()
        rand_state = np.random.get_state()
	if np.random.uniform()<= apply_prob:
    		psf_rms = np.random.uniform(low= 0.1 , high=my_max_psf_rms)
    		blurred_im = scipy.ndimage.filters.gaussian_filter( im.reshape(numpix_side,numpix_side) , psf_rms)
		if np.max(blurred_im)!=0:
    			blurred_im = blurred_im / np.max(blurred_im)
    		im[:] = blurred_im.reshape((-1,numpix_side*numpix_side))
	np.random.set_state(rand_state)


def add_poisson_noise(im,apply_prob=1):
	np.random.uniform()
	rand_state = np.random.get_state()
	if np.random.uniform()<= apply_prob:
		intensity_to_photoncounts = np.random.uniform(low=50.0, high=1000.0)
		photon_count_im = np.abs(im * intensity_to_photoncounts)
		poisson_noisy_im = np.random.poisson(lam=(photon_count_im), size=None)
		im_noisy = np.double(poisson_noisy_im)/intensity_to_photoncounts 
		im_noisy = im_noisy/np.max(im_noisy)
		im[:] = im_noisy
	np.random.set_state(rand_state)


def add_cosmic_ray(im,apply_prob=1):
	rand_state = np.random.get_state()
	if np.random.uniform()<= apply_prob:
		inds_cr = np.random.randint(0, high = 9999)
		filename_cr =  CRay_data_path + 'cosmicray_' + "%07d" % (inds_cr+1) + '.png'
		CR_MAP = np.array(Image.open(filename_cr),dtype='float32').reshape(numpix_side*numpix_side,)/255.0
		if np.max(CR_MAP)>0.1 and np.min(CR_MAP)<0.1:
			CR_MAP = CR_MAP/np.max(CR_MAP)
		else:
			CR_MAP = CR_MAP * 0
		CR_SCALE = np.random.uniform(low=0.0, high=max_cr_intensity)
		im[:] = im[:] + (CR_SCALE * CR_MAP)
	np.random.set_state(rand_state)

def pixellation(im_input):
        im = np.max(im_input)
        im =  im.reshape(numpix_side,numpix_side)
        numccdpix = np.random.randint(96, high=192)
        FACTOR = np.float( numccdpix)/192.0
        im_ccd =scipy.ndimage.interpolation.zoom( im , FACTOR )
        im_ccd_max = np.max(im_ccd)
        im_ccd = im_ccd * im_max / im_ccd_max
        add_gaussian_noise(im_ccd)
        im = scipy.ndimage.interpolation.zoom( im_ccd , 1/FACTOR )
        im = im * im_max / np.max(im)
	im_input[:] = im


def im_shift(im, m , n):
    shifted_im1 = np.zeros(im.shape)
    if n > 0:
        shifted_im1[n:,:] = im[:-n,:]
    elif n < 0:
        shifted_im1[:n,:] = im[-n:,:]
    elif n ==0:
        shifted_im1[:,:] = im[:,:]
    shifted_im2 = np.zeros(im.shape)
    if m > 0:
        shifted_im2[:,m:] = shifted_im1[:,:-m]
    elif m < 0:
        shifted_im2[:,:m] = shifted_im1[:,-m:]
    shifted_im2[np.isnan(shifted_im2)] = 0
    return shifted_im2

def pick_new_lens_center(ARCS,Y, xy_range = 0.5):
	rand_state = np.random.get_state()
	while True:
        	x_new = np.random.randint( -1 * np.ceil(xy_range/2/pix_res) , high = np.ceil(xy_range/2/pix_res) )
        	y_new = np.random.randint( -1 * np.ceil(xy_range/2/pix_res) , high = np.ceil(xy_range/2/pix_res) )
        	m_shift = - int(np.floor(Y[3]/pix_res) - x_new)
        	n_shift = - int(np.floor(Y[4]/pix_res) - y_new)
        	shifted_ARCS = im_shift(ARCS.reshape((numpix_side,numpix_side)), m_shift , n_shift ).reshape((numpix_side*numpix_side,))
		if np.sum(shifted_ARCS) >= ( 0.98 * np.sum(ARCS) ):
			break
        #lensXY = np.array( [ np.double(x_new) * pix_res+ (Y[3]%pix_res) , np.double(y_new) * pix_res + (Y[4]%pix_res) ])
	lensXY = np.array( [ np.double(m_shift) * pix_res+ Y[3] , np.double(n_shift) * pix_res + Y[4] ])
	np.random.set_state(rand_state)
	return shifted_ARCS , lensXY , m_shift, n_shift

def read_data_batch( X , Y , mag , max_file_num , train_or_test):
    batch_size = len(X)
    #mag = np.zeros((batch_size,1))
    if train_or_test=='test':
        #inds = range(batch_size)
        np.random.seed(seed=2)
	d_path = [[],[],[]]
	d_path[0] = test_data_path_1
	d_path[1] = test_data_path_2
	d_path[2] = test_data_path_3
	d_lens_path = [[],[],[]]
	d_lens_path[0] = testlens_data_path_1
	d_lens_path[1] = testlens_data_path_2
	d_lens_path[2] = testlens_data_path_3
        inds = np.random.randint(0, high = max_file_num , size= batch_size)
    else:
        np.random.seed(seed=None)
        inds = np.random.randint(0, high = max_file_num , size= batch_size)
	d_path = [[],[],[]]
        d_path[0] = arcs_data_path_1
        d_path[1] = arcs_data_path_2
	d_path[2] = arcs_data_path_3
        d_lens_path = [[],[],[]]
        d_lens_path[0] = lens_data_path_1
        d_lens_path[1] = lens_data_path_2
	d_lens_path[2] = lens_data_path_3

    #inds = np.zeros((batch_size,),dtype='int')
    for i in range(batch_size):

        #ARCS=1
        #nt = 0

	while True:
        	ARCS=1
        	nt = 0
        	while np.min(ARCS)==1 or np.max(ARCS)<0.4:
                	nt = nt + 1
			if nt>1:
				inds[i] = np.random.randint(0, high = max_file_num)



			pick_folder = np.random.randint(0, high = num_data_dirs)
			arc_filename = d_path[pick_folder] +  train_or_test + '_' + "%07d" % (inds[i]+1) + '.png'
			lens_filename = d_lens_path[pick_folder] +  train_or_test + '_' + "%07d" % (inds[i]+1) + '.png'

			if train_or_test=='test':
        			Y[i,:] = Y_all_test[pick_folder][inds[i],0:5]
				mag[i] = Y_all_test[pick_folder][inds[i],7]
    			else:
        			Y[i,:] = Y_all_train[pick_folder][inds[i],0:5]
				mag[i] = Y_all_train[pick_folder][inds[i],7]


                	ARCS = np.array(Image.open(arc_filename),dtype='float32').reshape(numpix_side*numpix_side,)/65535.0
			LENS = np.array(Image.open(lens_filename),dtype='float32').reshape(numpix_side*numpix_side,)/65535.0

		ARCS_SHIFTED, lensXY , m_shift, n_shift = pick_new_lens_center(ARCS,Y[i,:], xy_range = max_xy_range)
		LENS_SHIFTED = im_shift(LENS.reshape((numpix_side,numpix_side)), m_shift , n_shift ).reshape((numpix_side*numpix_side,))

		ARCS = np.copy(ARCS_SHIFTED) 
		Y[i,3] = lensXY[0]
		Y[i,4] = lensXY[1]


                if (np.all(np.isnan(ARCS)==False)) and ((np.all(ARCS>=0)) and (np.all(np.isnan(Y[i,3:5])==False))) and ~np.all(ARCS==0):
                        break

        rand_state = np.random.get_state()

	im_telescope = np.copy(ARCS) + LENS_SHIFTED * np.random.normal(loc=0.0, scale = 0.01) 
        apply_psf(im_telescope , max_psf_rms , apply_prob = 0.8)
	add_poisson_noise(im_telescope , apply_prob = 0.8)
	add_cosmic_ray(im_telescope,apply_prob = 0.8 )
	add_gaussian_noise(im_telescope)
	mask = gen_masks( 30 , ARCS.reshape((numpix_side,numpix_side)) , apply_prob = 0.5 )
	mask = 1.0



	
	if np.any(ARCS>0.4):
        	val_to_normalize = np.max(im_telescope[ARCS>0.4])
		if val_to_normalize==0:
			val_to_normalize = 1.0
		int_mult = np.random.normal(loc=1.0, scale = 0.01)
        	im_telescope = (im_telescope / val_to_normalize) * int_mult 


	im_telescope =  im_telescope.reshape(numpix_side,numpix_side)
        zero_bias = np.random.normal(loc=0.0, scale = 0.05)
        im_telescope = (im_telescope+zero_bias) * mask
        X[i,:] = im_telescope.reshape((1,-1))
	if np.any(np.isnan(X[i,:])) or np.any(np.isnan(Y[i,:])):
		X[i,:] = np.zeros((1,numpix_side*numpix_side))
		Y[i,:] = np.zeros((1,num_out))

	np.random.set_state(rand_state)
	#return 0

