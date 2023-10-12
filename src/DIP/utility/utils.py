import tensorflow as tf
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import config



@tf.function
def train_step_mse(sample, inj_model, optimizer_inj):
    """MSE training of the injective sub-network"""

    with tf.GradientTape() as tape:
        
        MSE = tf.keras.losses.MeanSquaredError()
        z , _ = inj_model(sample, reverse= False)
        recon = inj_model(z , reverse = True)[0]
        mse_loss = MSE(sample , recon)
        loss = mse_loss
        
        variables= tape.watched_variables()
        grads = tape.gradient(loss, variables)
        optimizer_inj.apply_gradients(zip(grads, variables))

    return loss


@tf.function
def train_step_ml(sample, bij_model, pz, optimizer_bij):
    """ML training of the bijective sub-network"""

    with tf.GradientTape() as tape:
        latent_sample, obj = bij_model(sample, reverse=False)
        p = -tf.reduce_mean(pz.prior.log_prob(latent_sample))
        j = -tf.reduce_mean(obj) # Log-det of Jacobian
        loss =  p + j
        variables = tape.watched_variables()
        grads = tape.gradient(loss, variables)
        optimizer_bij.apply_gradients(zip(grads, variables))

    return loss


def data_normalization(x):
    
    x = x.astype('float32')
    x = x - (x.max() + x.min())/2
    x /= (x.max())
    
    return x


def image_resizer(x , r):
    b , h, _ , nch = np.shape(x)
    y = np.zeros((np.shape(x)[0], r, r, nch))
    
    if x.shape[1] != r:
        for i in range(b):
            if nch == 1:
                y[i,:,:,0] = cv2.resize(x[i] , (r,r))
            else:
                y[i] = cv2.resize(x[i] , (r,r))         
    else:
        y = x
        
    return y
 

def PSNR(x_true , x_pred):
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += psnr(x_true[i],
                  x_pred[i],
                  data_range=x_true[i].max() - x_true[i].min())
        
    return s/np.shape(x_pred)[0]


def SSIM(x_true , x_pred):
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += ssim(x_true[i],
                  x_pred[i],
                  data_range=x_true[i].max() - x_true[i].min(),
                  multichannel=True)
        
    return s/np.shape(x_pred)[0]


def Dataset_preprocessing(dataset = 'MNIST', batch_size = 64):
    
    if dataset == 'mnist':

        (train_images, train_labels), (test_images, _) = tf.keras.datasets.mnist.load_data()
        if config.ood_experiment:

            np.random.seed(0)

            sorted_labels_ind = np.argsort(train_labels)
            sorted_labels = train_labels[sorted_labels_ind]
            test_ind = np.where(sorted_labels == 6)[0][0]

            train_images = train_images[sorted_labels_ind,:,:]
            test_images = train_images[test_ind:]
            train_images = train_images[:test_ind]
            np.random.shuffle(test_images)
            np.random.shuffle(train_images)



        train_images = np.expand_dims(train_images, axis = 3)
        test_images = np.expand_dims(test_images, axis = 3)
        
    elif dataset == 'ellipses':
        
        images = np.load('datasets/ellipses_64.npy')
        train_images , test_images = np.split(images , [55000])

    r = config.img_size
  
    train_images = image_resizer(train_images, r)
    test_images = image_resizer(test_images, r)
    train_images = data_normalization(train_images)
    test_images = data_normalization(test_images)
    
    train_images = data_normalization(train_images)
    test_images = data_normalization(test_images)
    
    train_images = tf.convert_to_tensor(train_images, tf.float32)
    test_images = tf.convert_to_tensor(test_images, tf.float32)
   
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images))
    
    SHUFFLE_BUFFER_SIZE = 256
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size,
                                                                     drop_remainder = True).prefetch(5)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset , test_dataset



def create_directory(path):
    """
    Creates a Directory with the given path. Throws a warning if it's already existing and an error if
    a file with the same name already exists.
    :param path: The full path to the new directory
    """
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IOError("Cannot create the output directory. There is a file with the same name: %s" % path)
        else:
            logging.debug("Directory already existing: %s" % path)
    else:
        try:
            os.makedirs(path)
        except OSError:
            raise IOError("Cannot create directory %s" % path)
    return 0



class Config:

    def __init__(self, cfg_file):
        import configparser as cfg
        # Parsing configuration file
        config = cfg.ConfigParser()
        config.read(cfg_file)

        # [TRAINING]
        section = 'TRAINING'
        self.train_injective = config.get(section, 'train_injective')
        self.train_bijective = config.get(section, 'train_bijective')
        self.n_epochs_inj = config.get(section, 'n_epochs_inj')
        self.n_epochs_bij = config.get(section, 'n_epochs_bij')
        self.img_size = config.get(section, 'img_size')
        self.batch_size = config.get(section, 'batch_size')
        self.dataset = config.get(section, 'dataset')
        self.lr = config.get(section, 'lr')
        self.gpu_num = config.get(section, 'gpu_num')
        self.desc = config.get(section, 'desc')
        self.inj_depth = config.get(section, 'inj_depth')
        self.bij_depth = config.get(section, 'bij_depth')
        self.reload = config.get(section, 'reload')
        self.ood_experiment = config.get(section, 'ood_experiment')
        self.unet_coupling = config.get(section, 'unet_coupling')
        self.n_test = config.get(section, 'n_test')

        # [ISPs]
        section = 'ISPs'
        self.inverse_scattering_solver = config.get(section, 'inverse_scattering_solver')

        # [MAP]
        section = 'MAP'
        self.run_map = config.get(section, 'run_map')
        self.reload_solver = config.get(section, 'reload_solver')
        self.problem_desc = config.get(section, 'problem_desc')
        self.noise_snr = config.get(section, 'noise_snr')
        self.er = config.get(section, 'er')
        self.solver = config.get(section, 'solver')
        self.optimizer = config.get(section, 'optimizer')
        self.lr_inv = config.get(section, 'lr_inv')
        self.initial_guess = config.get(section, 'initial_guess')
        self.nsteps = config.get(section, 'nsteps')
        self.scattering_data = config.get(section, 'scattering_data')
        self.fresnel_sample = config.get(section, 'fresnel_sample')
        self.cmap = config.get(section, 'cmap')
        self.tv_weight = config.get(section, 'tv_weight')
        self.num_objects = config.get(section, 'num_objects')

        # [POSTERIOR]
        section = 'POSTERIOR'
        self.run_posterior = config.get(section, 'run_posterior')
        self.reload_posterior = config.get(section, 'reload_posterior')
        self.nsteps_posterior = config.get(section, 'nsteps_posterior')
        self.lr_posterior = config.get(section, 'lr_posterior')
        self.test_nb = config.get(section, 'test_nb')
        self.beta = config.get(section, 'beta')