# Climate Simulation Project with Conditional Flow Generator

This project implements a conditional generator based on a WGAN-GP architecture for climate simulation. It includes:

- A training script using a combination of conditional flows, NLL loss, and optionally reconstruction loss.
- A visualization module to generate animations comparing predictions and ground truth.
- Tools to test and deploy the model via Hugging Face (interactive notebook, API, etc.).

---

## Table of Contents

- [Installation and Dependencies](#installation-and-dependencies)
- [Project Structure](#project-structure)
- [Running Training](#running-training)
- [Training Process](#training-process)
- [Testing a Model and Creating Visualizations](#testing-a-model-and-creating-visualizations)
- [Using Hugging Face](#using-hugging-face)

---

## Installation and Dependencies

Ensure you have Python 3.8 or higher installed. Install the required dependencies via pip:

```bash
pip install -r requirements.txt
```

If you plan to use [huggingface_hub](https://huggingface.co/docs/huggingface_hub) for model uploading or downloading, install it with:

```bash
pip install huggingface_hub
```

---

## Project Structure

- **`train.py`**  
  Training script handling data loading, model creation (conditional generator and WGAN-GP discriminator), training process, checkpointing, and logging via wandb.

- **`visu.py`**  
  Visualization module for generating animations and static visualizations (temperature, wind, differences, etc.).

- **`dataset.py`**  
  Contains `load_dataset` and `ERADataset` classes to load and preprocess the ERA5 dataset (or similar).

- **`models.py`**  
  Defines model architectures:  
  - `ConditionalFlowGenerator2d`  
  - `ConditionalWGANGPDiscriminator2d`  
  - `gradient_penalty_conditional`

- **`notebook_model_visualization.ipynb`**  
  Interactive notebook demonstrating how to download a checkpoint from Hugging Face Hub, load the model, generate predictions, and visualize results.

---

## Running Training

To start training, run the `train.py` script from the command line. For example:

```bash
python train.py --num_epochs 100 --batch_size 64 --lr 1e-4 --lr_discr 1e-4 --save_dir checkpoints --wandb_project ClimSim
```

### Main Arguments:

- `--save_dir`: Directory to save checkpoints.
- `--num_epochs`: Total number of training epochs.
- `--lr` and `--lr_discr`: Learning rates for generator and discriminator.
- `--batch_size`: Batch size.
- `--lambda_gp`: Gradient penalty coefficient.
- `--alpha_nll`: Weight of the NLL loss.
- `--use_recon` and `--alpha_recon`: Optionally activate and weight the reconstruction loss.
- `--nb_flows`: Number of flows in the generator.
- `--normalize`: Indicates whether to normalize data.
- `--wandb_project`: Project name for wandb logging.

The script automatically searches for a checkpoint in the specified directory and resumes training if one is found.

---

## Training Process

1. **Data Loading:**
   The `load_dataset` function splits the data files into training and validation sets. Data includes variables such as temperature and wind components, along with masks and geographical coordinates.

2. **Model Creation:**
   - The **generator** (`ConditionalFlowGenerator2d`) takes as input a tensor composed of inputs, masks, and coordinates.  
   - The **discriminator** (`ConditionalWGANGPDiscriminator2d`) receives both predictions and ground truth.

3. **Training:**
   - The generator produces samples via `sample` or `sample_most_probable`.
   - Discriminator loss is calculated with gradient penalty via `gradient_penalty_conditional`.
   - Generator training combines adversarial loss, NLL loss, and optionally reconstruction loss.
   - Logs (errors, mean and variance differences) are sent to wandb.
   - Checkpoints are saved at the end of each epoch.

---

## Testing a Model and Creating Visualizations

To test a model and generate visualizations (videos or images), use the visualization script (e.g., `visualize_final.py`):

```bash
python visualize_final.py --checkpoint checkpoints/checkpoint_epoch_10.pth --data_dir /path/to/era5_data --year 2000 --fps 24 --duration 10 --save_dir visualizations
```

### Visualization Features:

- **Loading the Model:**
  The `load_checkpoint_cf` function loads the conditional generator checkpoint and retrieves the number of flows used.

- **Prediction Generation:**
  The `sample_most_probable` method is used with `num_samples=100` to generate most likely predictions.

- **Video Generation:**
  The `compute_animation_for_scalar` function generates a temperature video after denormalization and conversion from Kelvin to Celsius.

---

## Using Hugging Face

The project includes an interactive notebook ([`notebook_model_visualization.ipynb`](./notebook_model_visualization.ipynb)) demonstrating how to:


**Download and Test a model on HF**
   - Use `hf_hub_download` to fetch the checkpoint.
   - Load the model and prepare predictions for visualizations.

---


