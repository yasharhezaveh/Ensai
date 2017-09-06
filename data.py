import numpy as np
from PIL import Image
import scipy.ndimage
from scipy import misc
from scipy.interpolate import RectBivariateSpline
from matplotlib import pyplot as plt

class data_processor:
    
    def __init__(self, arcs_data_paths, lens_data_paths,test_data_paths, testlens_data_paths, CRay_data_path, numpix_side, max_noise_rms, max_psf_rms, max_cr_intensity, variable_noise_rms, pix_res, min_unmasked_flux, num_data_dirs, max_xy_range, num_out):
        
        self.arcs_data_paths = arcs_data_paths
        self.lens_data_paths = lens_data_paths
        self.test_data_paths = test_data_paths
        self.testlens_data_paths = testlens_data_paths
        self.numpix_side = numpix_side
        self.max_noise_rms = max_noise_rms
        self.max_psf_rms = max_psf_rms
        self.max_cr_intensity = max_cr_intensity
        self.variable_noise_rms = variable_noise_rms
        self.pix_res = pix_res
        self.min_unmasked_flux = min_unmasked_flux
        self.num_data_dirs = num_data_dirs
        self.max_xy_range = max_xy_range
        self.num_out = num_out
        
        self.CRay_data_path = CRay_data_path
        

        Y_all_train = [np.loadtxt(arcs_data_path + '/parameters_train.txt' ) for arcs_data_path in arcs_data_paths]
        Y_all_test = [np.loadtxt(test_data_path + '/parameters_test.txt' ) for test_data_path in test_data_paths]
        R_n = np.loadtxt( 'data/PS_4_real.txt')
        I_n = np.loadtxt( 'data/PS_4_imag.txt')

        L_side = pix_res * numpix_side
        xv, yv = np.meshgrid( np.linspace(-L_side/2.0, L_side/2.0, num=numpix_side) ,  np.linspace(-L_side/2.0, L_side/2.0, num=numpix_side))

        self.L_side = L_side
        self.Y_all_train = Y_all_train
        self.Y_all_test = Y_all_test
        self.R_n = R_n
        self.I_n = I_n
        self.xv = xv
        self.yv = yv

    @staticmethod
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

    def read_batch_online(self, X , Y , max_file_num , train_or_test):
        num_samp = X.shape[0]
        Xmat, Ymat = eng.online_image_generator(num_samp , -1 , self.numpix_side , os.environ['LOCAL_SCRATCH'] , nargout=2)
        Xmat = np.array(Xmat._data.tolist())
        Xmat = Xmat.reshape((num_samp,self.numpix_side,self.numpix_side))
        Xmat = np.transpose(Xmat, axes=(0,2,1)).reshape((num_samp,self.numpix_side*self.numpix_side))
        Ymat = np.array(Ymat._data.tolist())
        Ymat = Ymat.reshape((self.num_out,num_samp)).transpose()
        X[:] = Xmat
        Y[:] = Ymat


    @staticmethod
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

    
    def add_gaussian_noise(self, im):
        if self.variable_noise_rms == False:
            rnd_noise_rms= self.max_noise_rms
        else:
            rnd_noise_rms = np.random.uniform(low=self.max_noise_rms/10, high=self.max_noise_rms)

        if np.random.uniform(low=0, high=1)<=0.25:
            noise_map = np.random.normal(loc=0.0, scale = rnd_noise_rms,size=im.shape)
        else:
            FFT_NOISE = np.random.normal(loc=0.0, scale = np.abs(self.R_n))  + np.random.normal(loc=0.0, scale = np.abs(self.I_n) ) *1j
            noise_map = self.make_real_noise(FFT_NOISE)
            noise_map = rnd_noise_rms * noise_map
            noise_map = noise_map.reshape((1,-1))
        im[:] = im[:] + noise_map



    def gen_masks(self, nmax,ARCS , apply_prob=0.5):
        mask = 1.0
        if np.min(ARCS)<0.1 and np.max(ARCS)>0.9:
            if np.random.uniform(low=0, high=1)<=apply_prob:
                while True:
                    mask = np.ones((self.numpix_side,self.numpix_side),dtype='float32')
                    num_mask = np.random.randint(1, high = nmax)
                    for j in range(num_mask):
                        x_mask =  np.random.uniform(low=-self.L_side/2.0, high=self.L_side/2.0)
                        y_mask =  np.random.uniform(low=-self.L_side/2.0, high=self.L_side/2.0)
                        r_mask = np.sqrt( (self.xv- x_mask  )**2 + (self.yv- y_mask )**2 )
                        mask_rad = 0.2
                        mask = mask * np.float32(r_mask>mask_rad)
                    if np.sum(mask*ARCS) >= ( self.min_unmasked_flux * np.sum(ARCS)):
                        break
        return mask


    def apply_psf(self, im , my_max_psf_rms , apply_prob=1.0 ):
        np.random.uniform()
        rand_state = np.random.get_state()
        if np.random.uniform()<= apply_prob:
            psf_rms = np.random.uniform(low= 0.1 , high=my_max_psf_rms)
            blurred_im = scipy.ndimage.filters.gaussian_filter( im.reshape(self.numpix_side,self.numpix_side) , psf_rms)
            if np.max(blurred_im)!=0:
                blurred_im = blurred_im / np.max(blurred_im)
            im[:] = blurred_im.reshape((-1,self.numpix_side*self.numpix_side))
        np.random.set_state(rand_state)

    @staticmethod
    def add_poisson_noise(im,apply_prob=1.0):
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


    def add_cosmic_ray(self,im,apply_prob=1):
        rand_state = np.random.get_state()
        if np.random.uniform()<= apply_prob:
            inds_cr = np.random.randint(0, high = 10000)
            filename_cr =  self.CRay_data_path + 'cosmicray_' + "%07d" % (inds_cr+1) + '.png'
            CR_MAP = np.array(Image.open(filename_cr),dtype='float32').reshape(self.numpix_side*self.numpix_side,)/255.0
            if np.max(CR_MAP)>0.1 and np.min(CR_MAP)<0.1:
                CR_MAP = CR_MAP/np.max(CR_MAP)
            else:
                CR_MAP = CR_MAP * 0
            CR_SCALE = np.random.uniform(low=0.0, high=self.max_cr_intensity)
            im[:] = im[:] + (CR_SCALE * CR_MAP)
        np.random.set_state(rand_state)

    def pixellation(self,im_input):
        im = np.max(im_input)
        im =  im.reshape(self.numpix_side,self.numpix_side)
        numccdpix = np.random.randint(96, high=192)
        FACTOR = np.float( numccdpix)/192.0
        im_ccd =scipy.ndimage.interpolation.zoom( im , FACTOR )
        im_ccd_max = np.max(im_ccd)
        im_ccd = im_ccd * im_max / im_ccd_max
        add_gaussian_noise(im_ccd)
        im = scipy.ndimage.interpolation.zoom( im_ccd , 1/FACTOR )
        im = im * im_max / np.max(im)
        im_input[:] = im

    @staticmethod
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

    def pick_new_lens_center(self, ARCS,Y, xy_range = 0.5):
        rand_state = np.random.get_state()
        while True:
            x_new = np.random.randint( -1 * np.ceil(xy_range/2/self.pix_res) , high = np.ceil(xy_range/2/self.pix_res) )
            y_new = np.random.randint( -1 * np.ceil(xy_range/2/self.pix_res) , high = np.ceil(xy_range/2/self.pix_res) )
            m_shift = - int(np.floor(Y[3]/self.pix_res) - x_new)
            n_shift = - int(np.floor(Y[4]/self.pix_res) - y_new)
            shifted_ARCS = self.im_shift(ARCS.reshape((self.numpix_side,self.numpix_side)), m_shift , n_shift ).reshape((self.numpix_side*self.numpix_side,))
            if np.sum(shifted_ARCS) >= ( 0.98 * np.sum(ARCS) ):
                break
        #lensXY = np.array( [ np.double(x_new) * pix_res+ (Y[3]%pix_res) , np.double(y_new) * pix_res + (Y[4]%pix_res) ])
        lensXY = np.array( [ np.double(m_shift) * self.pix_res+ Y[3] , np.double(n_shift) * self.pix_res + Y[4] ])
        np.random.set_state(rand_state)
        return shifted_ARCS , lensXY , m_shift, n_shift

    def read_data_batch(self, X , Y , mag , max_file_num , train_or_test):
        batch_size = len(X)
        #mag = np.zeros((batch_size,1))
        if train_or_test=='test':
            #inds = range(batch_size)
            np.random.seed(seed=2)
            d_path = self.test_data_paths
            d_lens_path = self.testlens_data_paths
            inds = np.random.randint(0, high = max_file_num , size= batch_size)
        else:
            np.random.seed(seed=None)
            inds = np.random.randint(0, high = max_file_num , size= batch_size)
            d_path = self.arcs_data_paths
            d_lens_path = self.lens_data_paths
        
        
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



                    pick_folder = np.random.randint(0, high = self.num_data_dirs)
                    arc_filename = d_path[pick_folder] +  train_or_test + '_' + "%07d" % (inds[i]+1) + '.png'
                    lens_filename = d_lens_path[pick_folder] +  train_or_test + '_' + "%07d" % (inds[i]+1) + '.png'

                    if train_or_test=='test':
                        Y[i,:] = self.Y_all_test[pick_folder][inds[i],0:5]
                        mag[i] = self.Y_all_test[pick_folder][inds[i],7]
                    else:
                        Y[i,:] = self.Y_all_train[pick_folder][inds[i],0:5]
                        mag[i] = self.Y_all_train[pick_folder][inds[i],7]


                    ARCS = np.array(Image.open(arc_filename),dtype='float32').reshape(self.numpix_side*self.numpix_side,)/65535.0
                    LENS = np.array(Image.open(lens_filename),dtype='float32').reshape(self.numpix_side*self.numpix_side,)/65535.0

                ARCS_SHIFTED, lensXY , m_shift, n_shift = self.pick_new_lens_center(ARCS,Y[i,:], xy_range = self.max_xy_range)
                LENS_SHIFTED = self.im_shift(LENS.reshape((self.numpix_side,self.numpix_side)), m_shift , n_shift ).reshape((self.numpix_side*self.numpix_side,))

                ARCS = np.copy(ARCS_SHIFTED) 
                Y[i,3] = lensXY[0]
                Y[i,4] = lensXY[1]


                if (np.all(np.isnan(ARCS)==False)) and ((np.all(ARCS>=0)) and (np.all(np.isnan(Y[i,3:5])==False))) and ~np.all(ARCS==0):
                    break

            rand_state = np.random.get_state()

            im_telescope = np.copy(ARCS) + LENS_SHIFTED * np.random.normal(loc=0.0, scale = 0.01)
            #print(im_telescope.shape)
            #plt.imshow(im_telescope[i].reshape(self.numpix_side,self.numpix_side),vmin=-0.15,vmax=1.0,cmap='nipy_spectral')
            self.apply_psf(im_telescope , self.max_psf_rms , apply_prob = 0.8)
            self.add_poisson_noise(im_telescope , apply_prob = 0.8)
            self.add_cosmic_ray(im_telescope,apply_prob = 0.8 )
            self.add_gaussian_noise(im_telescope)
            mask = self.gen_masks( 30 , ARCS.reshape((self.numpix_side,self.numpix_side)) , apply_prob = 0.5 )
            mask = 1.0



            if np.any(ARCS>0.4):
                val_to_normalize = np.max(im_telescope[ARCS>0.4])
                if val_to_normalize==0:
                    val_to_normalize = 1.0
                    int_mult = np.random.normal(loc=1.0, scale = 0.01)
                    im_telescope = (im_telescope / val_to_normalize) * int_mult 


            im_telescope =  im_telescope.reshape(self.numpix_side,self.numpix_side)
            zero_bias = np.random.normal(loc=0.0, scale = 0.05)
            im_telescope = (im_telescope+zero_bias) * mask
            X[i,:] = im_telescope.reshape((1,-1))
            if np.any(np.isnan(X[i,:])) or np.any(np.isnan(Y[i,:])):
                X[i,:] = np.zeros((1,self.numpix_side*self.numpix_side))
                Y[i,:] = np.zeros((1,self.num_out))

            np.random.set_state(rand_state)

        return X,Y,mag

