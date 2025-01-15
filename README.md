# Conditional WGAN-GP with Normalizing Flows

## Introduction

This project implements a **Conditional Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** integrated with **Normalizing Flows**. The model is designed to generate high-dimensional data conditioned on input variables, making it suitable for applications like climate simulations, traffic modeling, and other domains requiring complex data generation.

## Features

- **Conditional WGAN-GP**: Enhances the standard WGAN-GP by incorporating conditional information, allowing the generator to produce outputs based on specific input conditions.
- **Normalizing Flows**: Utilizes normalizing flows to model complex data distributions with exact likelihood computation.
- **TensorBoard Integration**: Logs training metrics and visualizations for real-time monitoring.
- **Checkpointing**: Saves model states after each epoch, enabling training resumption from saved checkpoints.
- **Gradient Clipping**: Stabilizes training by preventing gradient explosion.

## Project Structure

conditional-wgangp-flows/ │ ├── train_conditional_wgangp.py # Main training script ├── models.py # Model definitions (Generator, Discriminator, Flows) ├── dataset.py # Dataset handling and preprocessing ├── requirements.txt # Python dependencies ├── checkpoints/ # Directory for saving model checkpoints ├── runs/ # TensorBoard logs ├── README.md # Project documentation └── ... # Additional scripts and resources


## Model Architecture

### Generator (ConditionalFlowGenerator)

Combines a **Prior Network (PNet)** and multiple **Conditional Affine Flow Layers** to model the conditional distribution \( p(y|x) \):

- **PNet**: Processes input conditions \( x \) to produce mean and standard deviation for latent variables.
- **Flows**: Apply a series of affine transformations conditioned on \( x \), enabling complex distribution modeling.

### Discriminator (ConditionalWGANGPDiscriminator)

A **Conditional Discriminator** that evaluates the authenticity of generated samples \( y \) given the input conditions \( x \):

- **Input**: Concatenated \( x \) and \( y \) along the channel dimension.
- **Architecture**: Several convolutional layers followed by a linear layer to output a real-valued score.

## Training Procedure

1. **Initialization**: Instantiate the generator and discriminator models, and define their optimizers.
2. **Epoch Loop**: For each epoch:
   - **Discriminator Update**:
     - Generate fake samples using the generator.
     - Compute WGAN loss: \( \text{loss}_D = -\mathbb{E}[D(x, y_{\text{real}})] + \mathbb{E}[D(x, y_{\text{fake}})] \).
     - Apply Gradient Penalty to enforce Lipschitz constraint.
     - Backpropagate and update discriminator weights.
   - **Generator Update**:
     - Generate fake samples.
     - Compute adversarial loss: \( \text{loss}_{G_{\text{adv}}} = -\mathbb{E}[D(x, y_{\text{fake}})] \).
     - Compute Negative Log-Likelihood (NLL) loss.
     - Combine losses and backpropagate to update generator weights.
3. **Logging**: Record training metrics and generated samples to TensorBoard.
4. **Checkpointing**: Save model states after each epoch for future resumption.

## Logging and Checkpoints

### TensorBoard Logging

The training script logs the following metrics to TensorBoard:

- **Discriminator Loss**: Measures the discriminator's ability to distinguish real from fake samples.
- **Gradient Penalty**: Ensures the discriminator adheres to the Lipschitz constraint.
- **Generator Adversarial Loss**: Reflects the generator's performance in fooling the discriminator.
- **Generator NLL Loss**: Represents the likelihood of real data under the generator's distribution.
- **Generated Samples**: Visual comparisons between real and fake samples for qualitative assessment.

### Checkpointing

Model checkpoints are saved at the end of each epoch in the specified `save_dir`. Each checkpoint includes:

- **Generator State**: Saved as `gen_epoch_{epoch}.pth`.
- **Discriminator State**: Saved as `disc_epoch_{epoch}.pth`.

These checkpoints can be used to resume training or for later inference.

## Visualization

After each epoch (or at specified intervals), the training script generates and logs visual comparisons between real and fake samples:

- **Real Sample**: An example from the dataset.
- **Fake Sample**: Corresponding output from the generator.

These visualizations help in qualitatively assessing the generator's performance and ensuring that the generated data aligns well with real data patterns.
