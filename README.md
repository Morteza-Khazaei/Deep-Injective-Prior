# Deep Injective Prior for Inverse Scattering

This repository contains the official implementation for "Deep Injective Priors for Inverse Scattering". The project provides a framework for training a generative model composed of an injective and a bijective subnetwork, and then using this model as a prior to solve inverse scattering problems.

The core idea is to learn a manifold of realistic images and then constrain the solution of the inverse problem to lie on this learned manifold, leading to improved reconstructions, especially in noisy or ill-posed scenarios.

## Installation

1.  **Clone the repository and navigate into it:**
    ```bash
    git clone https://github.com/your-username/Deep-Injective-Prior.git
    cd Deep-Injective-Prior
    ```

2.  **Install the package:**
    This project is structured as an installable Python package. Installing it will handle all dependencies and make the `dip` command-line tool available in your environment. For development, it's best to install it in editable mode.

    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate

    # Install the package in editable mode
    pip install -e .
    ```
    Alternatively, you can install it for general use directly from a Git repository:
    ```bash
    pip install git+https://github.com/your-username/Deep-Injective-Prior.git
    ```

3.  **Download Datasets:**
    The MNIST dataset will be downloaded automatically by TensorFlow. For the `ellipses` dataset, ensure you have `datasets/ellipses_64.npy` available in the project root.

## Usage

After installation, all operations are performed using the `dip` command-line tool.

> **Note:** You must run the `dip` command from the root directory of the cloned repository, as it needs access to the `datasets` and `scattering_config` folders.

### 1. Training the Generative Model

The generative model consists of an injective and a bijective subnetwork. The most convenient way to train them is sequentially in a single command.

```bash
dip --train_injective --train_bijective --dataset ellipses --img_size 64 --n_epochs_inj 150 --n_epochs_bij 150 --batch_size 128 --lr 1e-4 --desc "ellipses_v1"
```

This command will:
1.  First, train the injective subnetwork for `n_epochs_inj` epochs.
2.  Then, using the in-memory weights of the just-trained injective model, it will train the bijective subnetwork for `n_epochs_bij` epochs.
3.  Checkpoints and results will be saved in `~/Desktop/DIP/experiments/ellipses_3_2_ellipses_v1/`.

#### Alternative: Separate Training

You can also run the training steps separately. This is useful if you want to fine-tune one part of the model later.

*   **a) Train the Injective Subnetwork:**
    ```bash
    dip --train_injective --dataset mnist --img_size 32 --n_epochs_inj 150 --lr 1e-4 --desc "mnist_training"
    ```

*   **b) Train the Bijective Subnetwork:**
    After the injective part is trained, train the bijective subnetwork. You **must** use the `--reload` flag to load the weights from the previous step.
    ```bash
    dip --train_bijective --dataset mnist --img_size 32 --n_epochs_bij 150 --lr 1e-4 --desc "mnist_training" --reload
    ```

### 2. Solving the Inverse Scattering Problem

Once the full generative model is trained, you can use it as a prior to solve an inverse scattering problem.

```bash
dip --inverse_scattering_solver --run_map --dataset mnist --desc "mnist_training" --reload --solver lso --initial_guess MOG --noise_snr 20 --nsteps 300 --lr_inv 5e-2
```

-   `--inverse_scattering_solver`: Flag to activate the solver.
-   `--run_map`: Perform MAP (Maximum A Posteriori) estimation.
-   `--reload`: Loads the trained generative model.
-   `--solver`: Choose the solver type. `lso` (Latent Space Optimization) is generally recommended.
-   `--initial_guess`: Choose the starting point for the optimization (`MOG` for Mixture of Gaussians or `BP` for Back-Propagation).
-   `--noise_snr`: The Signal-to-Noise Ratio (in dB) of the measurement noise.
-   `--nsteps`: Number of optimization steps.

### 3. Posterior Sampling

After finding the MAP estimate, you can sample from the posterior distribution to quantify uncertainty.

```bash
python -m src.dip.main --run_posterior --dataset mnist --desc "mnist_training" --reload --reload_solver --reload_posterior --noise_snr 20 --nsteps_posterior 10000 --beta 0.01
```

-   `--run_posterior`: Flag to run posterior sampling.
-   `--reload_solver`: Loads the result from the MAP estimation step.
-   `--beta`: A weight for the KL-divergence term in the loss function.

### Command-Line Arguments

For a full list of available options and their descriptions, run:

```bash
python -m src.dip.main --help
```

This will provide detailed information on all configurable parameters for training, solving, and posterior analysis.
