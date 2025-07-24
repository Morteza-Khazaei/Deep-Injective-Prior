# main.py - Updated for TensorFlow 2.18.0
import os
import sys
import cv2
import logging
import imageio
import numpy as np
from time import time


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
    parser = argparse.ArgumentParser(description="Generative Model Training and Posterior Modeling", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Training of the generative model
    parser.add_argument("--train_injective", help="Train the injective subnetwork", action="store_true", default=False)
    parser.add_argument("--train_bijective", help="Train the bijective subnetwork", action="store_true", default=False)
    parser.add_argument("-n_epochs_inj", help="Number of epochs for injective subnetwork", type=int, default=150)
    parser.add_argument("-n_epochs_bij", help="Number of epochs for bijective subnetwork", type=int, default=150)
    parser.add_argument("-img_size", help="Image size", type=int, default=32)
    parser.add_argument("-batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("-dataset", help="Dataset choice ('mnist' or 'ellipses')", type=str, default="mnist")
    parser.add_argument("-lr", help="Learning rate for network training", type=float, default=1e-4)
    parser.add_argument("-gpu_num", help="GPU selection", type=int, default=0)
    parser.add_argument("-desc", help="Experiment descriptor", type=str, default="default")
    parser.add_argument("-inj_depth", help="Injective network depth", type=int, default=3)
    parser.add_argument("-bij_depth", help="Bijective network depth", type=int, default=2)
    parser.add_argument("--reload", help="Reload existing trained network if it exists", action="store_true", default=False)
    parser.add_argument("--ood_experiment", help="Out-of-distribution experiment", action="store_true", default=False)
    parser.add_argument("--unet_coupling", help="Use U-Net as coupling layers", action="store_true", default=False)
    parser.add_argument("-n_test", help="Number of test samples for results", type=int, default=25)

    # Inverse Scattering Solver
    parser.add_argument("--inverse_scattering_solver", help="Run inverse scattering solver", action="store_true", default=False)
    parser.add_argument("--run_map", help="Run MAP solver", action="store_true", default=False)
    parser.add_argument("--reload_solver", help="Reload existing solver if it exists", action="store_true", default=False)
    parser.add_argument("-problem_desc", help="Experiment descriptor for solver", type=str, default="default")
    parser.add_argument("-noise_snr", help="Noise SNR (dB)", type=float, default=30)
    parser.add_argument("-er", help="Maximum epsilon_r of the medium", type=float, default=2.0)
    parser.add_argument("-solver", help="Solver choice ('lso' or 'dso')", type=str, default="lso")
    parser.add_argument("-optimizer", help="Optimizer choice ('Adam' or 'lbfgs')", type=str, default="Adam")
    parser.add_argument("-lr_inv", help="Learning rate of inverse scattering solver for Adam", type=float, default=5e-2)
    parser.add_argument("-initial_guess", help="Initial guess ('BP' or 'MOG')", type=str, default="MOG")
    parser.add_argument("-nsteps", help="Number of optimization steps", type=int, default=300)
    parser.add_argument("-scattering_data", help="Scattering data choice ('synthetic' or 'real')", type=str, default="synthetic")
    parser.add_argument("-fresnel_sample", help="Fresnel sample choice", type=str, default="FoamDielExt")
    parser.add_argument("-cmap", help="Color map", type=str, default="seismic")
    parser.add_argument("-tv_weight", help="TV multiplier", type=float, default=0.0)
    parser.add_argument("-num_objects", help="Number of test samples for solver", type=int, default=1)

    # Posterior Modeling
    parser.add_argument("--run_posterior", help="Run posterior sampling", action="store_true", default=False)
    parser.add_argument("--reload_posterior", help="Reload existing posterior model if it exists", action="store_true", default=False)
    parser.add_argument("-nsteps_posterior", help="Number of posterior optimization steps", type=int, default=10000)
    parser.add_argument("-lr_posterior", help="Learning rate of posterior optimizer", type=float, default=1e-2)
    parser.add_argument("-test_nb", help="Test sample number for posterior modeling", type=int, default=0)
    parser.add_argument("-beta", help="KL multiplier", type=float, default=0.01)
    parser.add_argument("-v", "--verbose", help="Provides detailed (DEBUG) logging for DIP. Default is false", default=False, action="store_true")

    args = parser.parse_args()


    # TODO Add error skipping
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logger = init_loggers(msg_level=logging_level)

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
                logger.debug("Directory already existing: %s" % path)
        else:
            try:
                os.makedirs(path)
            except OSError:
                raise IOError("Cannot create directory %s" % path)
        return 0

    workspace = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop', 'DIP')
    create_directory(workspace)

    # change working directory
    os.chdir(workspace)
    logger.info(f'Current working directory: {os.getcwd()}')

    # It is crucial to configure the GPU before TensorFlow is initialized.
    # Importing TensorFlow (or any module that imports it, like the project's models or utils)
    # will initialize the TF runtime. By setting the visible devices first, we ensure
    # TF only sees and uses the specified GPU.
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    logger.info(f"Found GPUs: {gpus}")
    if gpus:
        # Restrict TensorFlow to only use the selected GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[args.gpu_num], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[args.gpu_num], True)
            logger.info(f"Successfully set GPU {args.gpu_num} as visible.")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            logger.debug(e)
            logger.warning(f"Could not set GPU {args.gpu_num}. It might already be in use or initialized.")

    # Now that the GPU is configured, we can import the rest of the project modules
    # that depend on TensorFlow.
    from .utility import utils, scattering_utils
    from .models import injective, bijective, prior
    import matplotlib.pyplot as plt

    # experiment path
    exp_path = os.path.join('experiments', f'{args.dataset}_{args.inj_depth}_{args.bij_depth}_{args.desc}')
    create_directory(exp_path)

    train_dataset, test_dataset = utils.Dataset_preprocessing(
        dataset=args.dataset, img_size=args.img_size, batch_size=args.batch_size, ood_experiment=args.ood_experiment
    )
    logger.info(f"Dataset is loaded: training and test dataset shape: {np.shape(next(iter(train_dataset)))}, {np.shape(next(iter(test_dataset)))}")

    _ , image_size , _ , c = np.shape(next(iter(train_dataset)))
    latent_dim = 64

    optimizer_inj = tf.keras.optimizers.Adam(learning_rate=args.lr) # Optimizer of injective sub-network
    optimizer_bij = tf.keras.optimizers.Adam(learning_rate=args.lr) # Optimizer of bijective sub-network

    pz = prior(latent_dim=latent_dim)
    inj_model = injective(revnet_depth=args.inj_depth, image_size=image_size) # Injective network
    bij_model = bijective(revnet_depth=args.bij_depth) # Bijective network

    # call generator once to set weights (Data dependent initialization for act norm layer)
    dummy_x = next(iter(train_dataset))
    dummy_z, _ = inj_model(dummy_x, reverse=False)
    dummy_l_z , _ = bij_model(dummy_z, reverse=False)

    num_params_inj_model = np.sum([np.prod(v.shape) for v in inj_model.trainable_weights])
    num_params_bij_model = np.sum([np.prod(v.shape) for v in bij_model.trainable_weights])
    logger.info('Number of trainable parameters of injective subnetwork: {}'.format(num_params_inj_model))
    logger.info('Number of trainable parameters of bijective subnetwork: {}'.format(num_params_bij_model))

    ckpt = tf.train.Checkpoint(pz=pz , inj_model=inj_model, optimizer_inj=optimizer_inj,
        bij_model=bij_model, optimizer_bij=optimizer_bij)
    manager = tf.train.CheckpointManager(
        ckpt, os.path.join(exp_path, 'checkpoints'), max_to_keep=5)

    if args.reload:
        status = ckpt.restore(manager.latest_checkpoint)
        logger.info("Restored from {}".format(status))
    else:
        logger.info("Initializing from scratch.")

    samples_folder = os.path.join(exp_path, 'Results')
    os.makedirs(samples_folder, exist_ok=True)

    if args.train_injective:
        # print an empty line in logger
        logger.info('')

        logger.info('Training of the Injective Subnetwork')
        logger.info('---> Dataset: {}'.format(args.dataset))
        logger.info('---> Experiment path: {}'.format(exp_path))
        logger.info('---> Injective depth: {}'.format(args.inj_depth))
        logger.info('---> Num epochs: {}'.format(args.n_epochs_inj))
        logger.info('---> Learning rate: {}'.format(args.lr))

        ngrid = int(np.sqrt(args.n_test))
        image_path_reconstructions = os.path.join(
            samples_folder, 'Reconstructions')

        def save_image_grid(image_tensor, path, ngrid, image_size, c):
            """Reshapes and saves a batch of images as a grid."""
            # Denormalize from [-1, 1] to [0, 255]
            image_array = (image_tensor.numpy() * 127.5 + 127.5)
            
            # Reshape into a grid
            image_array = image_array[:, :, :, ::-1].reshape(
                ngrid, ngrid, image_size, image_size, c
            ).swapaxes(1, 2).reshape(ngrid * image_size, -1, c)
            
            # Clip and save
            image_array = image_array.clip(0, 255).astype(np.uint8)
            imageio.imwrite(path, image_array[:, :, 0])

        @tf.function
        def evaluation_step(inj_model, test_batch):
            """Performs evaluation on a test batch and returns metrics on the GPU."""
            z_test, _ = inj_model(test_batch, reverse=False, training=False)
            test_recon, _ = inj_model(z_test, reverse=True, training=False)
            # PSNR for data in [-1, 1] range has a max_val of 2.0
            psnr = tf.image.psnr(test_batch, test_recon, max_val=2.0)
            return tf.reduce_mean(psnr), test_recon

        os.makedirs(image_path_reconstructions, exist_ok=True)

        # Fetch one batch of test data outside the loop for consistent evaluation
        test_gt = next(iter(test_dataset))[:args.n_test]

        for epoch in range(args.n_epochs_inj):
            epoch_start = time()  
            for x in train_dataset:
                utils.train_step_mse(x, inj_model, optimizer_inj)

            # --- Evaluation and Logging ---
            psnr, test_recon = evaluation_step(inj_model, test_gt)

            # Save images less frequently to avoid I/O bottlenecks
            if epoch % 5 == 0 or epoch == args.n_epochs_inj - 1:
                save_image_grid(test_recon, os.path.join(image_path_reconstructions, f'{epoch}_recon.png'), ngrid, image_size, c)
                save_image_grid(test_gt, os.path.join(image_path_reconstructions, f'{epoch}_gt.png'), ngrid, image_size, c)
            
            epoch_end = time()
            ellapsed_time = epoch_end - epoch_start
            logger.info("Epoch: {:03d}| time: {:.0f}| PSNR: {:.3f}"
                    .format(epoch, ellapsed_time, psnr.numpy()))
                
            with open(os.path.join(exp_path, 'results.txt'), 'a') as f:
                f.write("Epoch: {:03d}| time: {:.0f}| PSNR: {:.3f}"
                    .format(epoch, ellapsed_time, psnr))
                f.write('\n')
            
            manager.save()

    if args.train_bijective:
        # print an empty line in logger
        logger.info('')

        logger.info('Training of the Bijective Subnetwork:')
        logger.info('---> Dataset: {}'.format(args.dataset))
        logger.info('---> Experiment path: {}'.format(exp_path))
        logger.info('---> Bijective depth: {}'.format(args.bij_depth))
        logger.info('---> Num epochs: {}'.format(args.n_epochs_bij))
        logger.info('---> Learning rate: {}'.format(args.lr))

        ngrid = int(np.sqrt(args.n_test))
        image_path_generated = os.path.join(samples_folder, 'Generated samples')
        os.makedirs(image_path_generated, exist_ok=True)

        @tf.function
        def get_latent_representation(x):
            """Applies the injective model to a batch of images to get its latent representation."""
            z, _ = inj_model(x, reverse=False, training=False)
            return z

        # Create a new dataset by applying the injective model to the training data on-the-fly.
        # This leverages tf.data for efficiency, avoiding high memory usage and CPU-GPU data transfers.
        z_inters_dataset = train_dataset.map(get_latent_representation, num_parallel_calls=tf.data.AUTOTUNE)
        z_inters_dataset = z_inters_dataset.prefetch(tf.data.AUTOTUNE)

        @tf.function
        def generate_samples_step(pz, bij_model, inj_model, n_samples):
            """Generates samples from the prior and passes them through the networks."""
            z_base = pz.prior.sample(n_samples)
            z_inter, _ = bij_model(z_base, reverse=True, training=False)
            generated_samples, _ = inj_model(z_inter, reverse=True, training=False)
            return generated_samples

        for epoch in range(args.n_epochs_bij):
            epoch_start = time()
            # The loss from the last batch will be used for logging
            for x in z_inters_dataset:
                ml_loss = utils.train_step_ml(x, bij_model, pz, optimizer_bij)
                        
            # Sampling and saving (less frequently to avoid I/O bottlenecks)
            if epoch % 5 == 0 or epoch == args.n_epochs_bij - 1:
                generated_samples = generate_samples_step(pz, bij_model, inj_model, args.n_test)
                save_image_grid(generated_samples, os.path.join(image_path_generated, f'{epoch}_samples.png'), ngrid, image_size, c)

            epoch_end = time()       
            ellapsed_time = epoch_end - epoch_start
            logger.info("Epoch: {:03d}| time: {:.0f}| ML Loss: {:.3f}"
                    .format(epoch, ellapsed_time, ml_loss.numpy()))
                
            with open(os.path.join(exp_path, 'results.txt'), 'a') as f:
                f.write("Epoch: {:03d}| time: {:.0f}| ML Loss: {:.3f}"
                    .format(epoch, ellapsed_time, ml_loss))
                f.write('\n')
            
            manager.save()


    if args.inverse_scattering_solver:
        # print an empty line in logger
        logger.info('')

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
        scattering_op = scattering_utils.scattering_op(args, n_inc_wave=12)

        scattering_pipeline = scattering_utils.scattering_solver(args, exp_path, scattering_op, inj_model, bij_model, pz=pz)

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
                MAP_estimate = scattering_pipeline.MAP_estimator(measurements, testing_images, lam=0) 
            elif args.solver == 'dso':
                MAP_estimate = scattering_pipeline.MAP_estimator(measurements, testing_images, lam=1e-2) 

        if args.run_posterior:
            scattering_pipeline.posterior_sampling(measurements[args.test_nb:args.test_nb+1], testing_images[args.test_nb:args.test_nb+1])

if __name__ == "__main__":
    main()