import os
import sys
import cv2
import logging
import numpy as np
from time import time
import tensorflow as tf
from utility import utils as ut
from models import injective, bijective, prior
from utility import scattering_utils
import imageio

import matplotlib.pyplot as plt


@staticmethod
def init_loggers(msg_level=logging.DEBUG):
    """
    Init a stdout logger
    :param msg_level: Standard msgLevel for both loggers. Default is DEBUG
    """

    logging.getLogger().addHandler(logging.NullHandler())
    # Create default path or get the pathname without the extension, if there is one
    dip_logger = logging.getLogger("root")
    dip_logger.handlers = []  # Remove the standard handler again - Bug in logging module

    dip_logger.setLevel(msg_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    dip_logger.addHandler(console_handler)
    
    return dip_logger


def main():
    assert sys.version_info >= (3, 5), "DIP needs python >= 3.5.\n Run 'python --version' for more info."
    import argparse
    parser = argparse.ArgumentParser(description="Generative Model Training and Posterior Modeling")

    # Training of the generative model
    parser.add_argument("--train_injective", help="Train the injective subnetwork", action="store_true", default=True)
    parser.add_argument("--train_bijective", help="Train the bijective subnetwork", action="store_true", default=True)
    parser.add_argument("--n_epochs_inj", help="Number of epochs for injective subnetwork", type=int, default=150)
    parser.add_argument("--n_epochs_bij", help="Number of epochs for bijective subnetwork", type=int, default=150)
    parser.add_argument("--img_size", help="Image size", type=int, default=32)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("--dataset", help="Dataset choice ('mnist' or 'ellipses')", type=str, default="mnist")
    parser.add_argument("--lr", help="Learning rate for network training", type=float, default=1e-4)
    parser.add_argument("--gpu_num", help="GPU selection", type=int, default=0)
    parser.add_argument("--desc", help="Experiment descriptor", type=str, default="default")
    parser.add_argument("--inj_depth", help="Injective network depth", type=int, default=3)
    parser.add_argument("--bij_depth", help="Bijective network depth", type=int, default=2)
    parser.add_argument("--reload", help="Reload existing trained network if it exists", action="store_true", default=True)
    parser.add_argument("--ood_experiment", help="Out-of-distribution experiment", action="store_true", default=False)
    parser.add_argument("--unet_coupling", help="Use U-Net as coupling layers", action="store_true", default=True)
    parser.add_argument("--n_test", help="Number of test samples for results", type=int, default=25)

    # Inverse Scattering Solver
    parser.add_argument("--inverse_scattering_solver", help="Run inverse scattering solver", action="store_true", default=True)
    parser.add_argument("--run_map", help="Run MAP solver", action="store_true", default=True)
    parser.add_argument("--reload_solver", help="Reload existing solver if it exists", action="store_true", default=True)
    parser.add_argument("--problem_desc", help="Experiment descriptor for solver", type=str, default="default")
    parser.add_argument("--noise_snr", help="Noise SNR (dB)", type=float, default=30)
    parser.add_argument("--er", help="Maximum epsilon_r of the medium", type=float, default=2.0)
    parser.add_argument("--solver", help="Solver choice ('lso' or 'dso')", type=str, default="lso")
    parser.add_argument("--optimizer", help="Optimizer choice ('Adam' or 'lbfgs')", type=str, default="Adam")
    parser.add_argument("--lr_inv", help="Learning rate of inverse scattering solver for Adam", type=float, default=5e-2)
    parser.add_argument("--initial_guess", help="Initial guess ('BP' or 'MOG')", type=str, default="MOG")
    parser.add_argument("--nsteps", help="Number of optimization steps", type=int, default=300)
    parser.add_argument("--scattering_data", help="Scattering data choice ('synthetic' or 'real')", type=str, default="synthetic")
    parser.add_argument("--fresnel_sample", help="Fresnel sample choice", type=str, default="FoamDielExt")
    parser.add_argument("--cmap", help="Color map", type=str, default="seismic")
    parser.add_argument("--tv_weight", help="TV multiplier", type=float, default=0.0)
    parser.add_argument("--num_objects", help="Number of test samples for solver", type=int, default=1)

    # Posterior Modeling
    parser.add_argument("--run_posterior", help="Run posterior sampling", action="store_true", default=True)
    parser.add_argument("--reload_posterior", help="Reload existing posterior model if it exists", action="store_true", default=False)
    parser.add_argument("--nsteps_posterior", help="Number of posterior optimization steps", type=int, default=10000)
    parser.add_argument("--lr_posterior", help="Learning rate of posterior optimizer", type=float, default=1e-2)
    parser.add_argument("--test_nb", help="Test sample number for posterior modeling", type=int, default=0)
    parser.add_argument("--beta", help="KL multiplier", type=float, default=0.01)
    parser.add_argument("-v", "--verbose", help="Provides detailed (DEBUG) logging for DIP. Default is false", default=False, action="store_true")

    args = parser.parse_args()


    # TODO Add error skipping
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logger = init_loggers(msg_level=logging_level)

    workspace = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop', 'DIP')
    ut.create_directory(workspace)

    all_experiments = os.path.join(workspace, 'experiments')
    ut.create_directory(all_experiments)

    # experiment path
    exp_path = os.path.join(all_experiments, f'{args.dataset}_{args.inj_depth}_{args.bij_depth}_{args.desc}')
    ut.create_directory(exp_path)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[args.gpu_num], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[args.gpu_num], True)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            logger.debug(e)

    train_dataset , test_dataset = ut.Dataset_preprocessing(dataset=args.dataset, batch_size=args.batch_size)
    logger.info(f'Dataset is loaded: training and test dataset shape: 
        {np.shape(next(iter(train_dataset)))} {np.shape(next(iter(test_dataset)))}')

    _ , image_size , _ , c = np.shape(next(iter(train_dataset)))
    latent_dim = 64

    optimizer_inj = tf.keras.optimizers.Adam(learning_rate=args.lr) # Optimizer of injective sub-network
    optimizer_bij = tf.keras.optimizers.Adam(learning_rate=args.lr) # Optimizer of bijective sub-network

    pz = prior(latent_dim = latent_dim)
    inj_model = injective(revnet_depth = args.inj_depth,
                        image_size = image_size) # Injective network

    bij_model = bijective(revnet_depth = args.bij_depth) # Bijective network

    num_params_inj_model = np.sum([np.prod(v.get_shape()) for v in inj_model.trainable_weights])
    num_params_bij_model = np.sum([np.prod(v.get_shape()) for v in bij_model.trainable_weights])
    logger.info('Number of trainable parameters of injective subnetwork: {}'.format(num_params_inj_model))
    logger.info('Number of trainable parameters of bijective subnetwork: {}'.format(num_params_bij_model))

    # call generator once to set weights (Data dependent initialization for act norm layer)
    dummy_x = next(iter(train_dataset))
    dummy_z, _ = inj_model(dummy_x, reverse=False)
    dummy_l_z , _ = bij_model(dummy_z, reverse=False)

    ckpt = tf.train.Checkpoint(pz = pz , inj_model= inj_model, optimizer_inj=optimizer_inj,
        bij_model=bij_model, optimizer_bij= optimizer_bij)
    manager = tf.train.CheckpointManager(
        ckpt, os.path.join(exp_path, 'checkpoints'), max_to_keep=5)

    if args.reload:
        ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint and args.reload:
        logger.info("Restored from {}".format(manager.latest_checkpoint))
    else:
        logger.info("Initializing from scratch.")

    samples_folder = os.path.join(exp_path, 'Results')
    os.makedirs(samples_folder, exist_ok=True)

    if args.train_injective:

        logger.info('Training of the Injective Subnetwork')
        logger.info('---> Dataset: {}'.format(args.dataset))
        logger.info('---> Experiment path: {}'.format(exp_path))
        logger.info('---> Injective depth: {}'.format(args.inj_depth))
        logger.info('---> Num epochs: {}'.format(args.n_epochs_inj))
        logger.info('---> Learning rate: {}'.format(args.lr))

        ngrid = int(np.sqrt(args.n_test))
        image_path_reconstructions = os.path.join(
            samples_folder, 'Reconstructions')

        os.makedirs(image_path_reconstructions, exist_ok=True)

        for epoch in range(args.n_epochs_inj):
            epoch_start = time()  
            for x in train_dataset:
                ut.train_step_mse(x, inj_model, optimizer_inj)
            
            # Reconstrctions
            test_gt = next(iter(test_dataset))[:args.n_test]
            z_test = inj_model(test_gt[:args.n_test], reverse= False)[0] 
            test_recon = inj_model(z_test , reverse = True)[0].numpy()[:args.n_test]
            psnr = ut.PSNR(test_gt.numpy(), test_recon)

            test_recon = test_recon[:, :, :, ::-1].reshape(ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
                c)*127.5 + 127.5
            test_recon = test_recon.clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(image_path_reconstructions, '%d_recon.png' % (epoch,)),
                test_recon[:,:,0]) # Reconstructed test images
            
            test_gt = test_gt.numpy()[:, :, :, ::-1].reshape(ngrid, ngrid,image_size,
                image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)* 127.5 + 127.5
            test_gt = test_gt.clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(image_path_reconstructions, '%d_gt.png' % (epoch,)),
                test_gt[:,:,0]) # Ground truth test images
            
            epoch_end = time()       
            ellapsed_time = epoch_end - epoch_start
            logger.info("Epoch: {:03d}| time: {:.0f}| PSNR: {:.3f}"
                    .format(epoch, ellapsed_time, psnr))
                
            with open(os.path.join(exp_path, 'results.txt'), 'a') as f:
                f.write("Epoch: {:03d}| time: {:.0f}| PSNR: {:.3f}"
                    .format(epoch, ellapsed_time, psnr))
                f.write('\n')
            
            save_path = manager.save()


    if args.train_bijective:

        logger.info('Training of the Bijective Subnetwork:')
        logger.info('---> Dataset: {}'.format(args.dataset))
        logger.info('---> Experiment path: {}'.format(exp_path))
        logger.info('---> Bijective depth: {}'.format(args.bij_depth))
        logger.info('---> Num epochs: {}'.format(args.n_epochs_bij))
        logger.info('---> Learning rate: {}'.format(args.lr))

        ngrid = int(np.sqrt(args.n_test))
        image_path_generated = os.path.join(samples_folder, 'Generated samples')
        os.makedirs(image_path_generated, exist_ok=True)

        z_inters = np.zeros([len(list(train_dataset)) * args.batch_size , latent_dim])
        cnt = 0
        for x in train_dataset:
            z_inter, _ = inj_model(x, reverse = False)
            z_inters[cnt*args.batch_size:(cnt+1)*args.batch_size] = z_inter.numpy()
            cnt = cnt + 1

        z_inters = tf.convert_to_tensor(z_inters, tf.float32)
        z_inters_dataset = tf.data.Dataset.from_tensor_slices((z_inters))
        z_inters_dataset = z_inters_dataset.shuffle(args.batch_size * 3).batch(args.batch_size , drop_remainder = True).prefetch(5)
                
        for epoch in range(args.n_epochs_bij):
            epoch_start = time()
            for x in z_inters_dataset:
                ml_loss = ut.train_step_ml(x, bij_model, pz, optimizer_bij).numpy()
                        
            # Sampling
            z_base = pz.prior.sample(args.n_test) # sampling from base (Gaussian) 
            z_inter = bij_model(z_base , reverse = True)[0] # Intermediate samples 
            generated_samples = inj_model(z_inter , reverse = True)[0].numpy() # Randmly generated samples
            
            generated_samples = generated_samples[:, :, :, ::-1].reshape(ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
                c)*127.5 + 127.5
            generated_samples = generated_samples.clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(image_path_generated, '%d_samples.png' % (epoch,)),
                generated_samples[:,:,0]) # Generated samples

            epoch_end = time()       
            ellapsed_time = epoch_end - epoch_start
            logger.info("Epoch: {:03d}| time: {:.0f}| ML Loss: {:.3f}"
                    .format(epoch, ellapsed_time, ml_loss))
                
            with open(os.path.join(exp_path, 'results.txt'), 'a') as f:
                f.write("Epoch: {:03d}| time: {:.0f}| ML Loss: {:.3f}"
                    .format(epoch, ellapsed_time, ml_loss))
                f.write('\n')
            
            save_path = manager.save()


    if args.inverse_scattering_solver:

        logger.info('Solving Inverse Scattering:')
        logger.info('---> Dataset: {}'.format(args.dataset))
        logger.info('---> Experiment path: {}'.format(exp_path))
        logger.info('---> Epsilon_r: {}'.format(args.er))
        logger.info('---> Noise snr: {}'.format(args.noise_snr))
        logger.info('---> Solver:{}'.format(args.solver))
        logger.info('---> Initial guess: {}'.format(args.initial_guess))
        logger.info('---> Optimizer: {}'.format(args.optimizer))
        logger.info('---> Solver learning rate:{}'.format(args.lr_inv))

        n_test = 1 if args.scattering_data == 'real' else args.num_objects
        testing_images = next(iter(test_dataset))[:n_test]
        scattering_op = scattering_utils.scattering_op(n_inc_wave = 12)

        scattering_pipeline = scattering_utils.scattering_solver(exp_path, scattering_op, inj_model, bij_model, pz= pz)

        if args.scattering_data == 'real':
            if args.fresnel_sample == 'FoamDielExt':
                setup = np.load('scattering_args/FoamDielExt.npz')
            elif args.fresnel_sample == 'FoamTwinDiel':
                setup = np.load('scattering_args/FoamTwinDiel.npz')

            testing_images = setup['gt']
            testing_images = cv2.resize(testing_images , (args.img_size,args.img_size))
            testing_images = testing_images[None,...][...,None]
            testing_images_write = (args.er-1) * ((testing_images + 1)/2) + 1

            gt_path = os.path.join(scattering_pipeline.prob_folder, 'gt.png')
            plt.imshow(testing_images_write[0,:,:,0], cmap = args.cmap)
            plt.colorbar()
            plt.savefig(gt_path)
            plt.close()
            testing_images = tf.convert_to_tensor(testing_images)
            Es = setup['Es']
            measurements = tf.convert_to_tensor(Es[None,...], tf.complex64)

        else:
            measurements = scattering_pipeline.forward_solver(testing_images)

        if args.run_map:
            if args.solver == 'lso':
                MAP_estimate = scattering_pipeline.MAP_estimator(measurements, testing_images , lam=0) 
            elif args.solver == 'dso':
                MAP_estimate = scattering_pipeline.MAP_estimator(measurements, testing_images, lam=1e-2) 

        if args.run_posterior:
            scattering_pipeline.posterior_sampling(measurements[args.test_nb:args.test_nb+1], testing_images[args.test_nb:args.test_nb+1])