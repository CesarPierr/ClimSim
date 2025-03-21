import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
import json
import os
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import load_dataset, ERADataset

# Render settings
mpl.rcParams['text.antialiased'] = True
mpl.rcParams['lines.antialiased'] = True
mpl.use('Agg')

# =============================================================================
# Helper Functions
# =============================================================================

def setup_cartopy_axis(ax):
    """Add common static map features to an axis."""
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    return ax

class TQDMWriter(FFMpegWriter):
    """Custom FFMpegWriter that wraps a tqdm progress bar during rendering."""
    def __init__(self, total_frames, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pbar = tqdm(total=total_frames, desc="Rendering frames", unit="frame")

    def grab_frame(self, **savefig_kwargs):
        super().grab_frame(**savefig_kwargs)
        self._pbar.update(1)

    def finish(self):
        super().finish()
        self._pbar.close()

def get_animation_writer(save_path, fps, total_frames=None):
    """Return the appropriate animation writer.
       If total_frames is provided and ffmpeg is available, use TQDMWriter.
    """
    if save_path.endswith('.mp4'):
        if animation.writers['ffmpeg'].isAvailable():
            if total_frames is not None:
                writer = TQDMWriter(total_frames=total_frames, fps=fps, metadata=dict(artist='Me'), bitrate=5000)
            else:
                writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=5000)
        else:
            print("FFmpeg non disponible. Passage au format GIF.")
            save_path = save_path.replace('.mp4', '.gif')
            writer = PillowWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)
    return writer, save_path

# =============================================================================
# Model and Prediction Functions
# =============================================================================

def load_checkpoint(run_name, device=torch.device('cpu'), type='best', checkpoints='./checkpoints'):
    checkpoints = Path(checkpoints)
    with open('experiments.json', 'r') as file:
        configs = json.load(file)
    # Find configuration matching run_name
    idx = [i for i, cfg in enumerate(configs) if cfg['experiment_name'] == run_name][0]
    config = configs[idx]
    file_name = run_name + f'_{type}.pt'
    checkpoint_path = checkpoints / config['model'] / file_name
    checkpoint = torch.load(checkpoint_path)

    # Dynamic model import
    ClimatePINN = getattr(__import__(f'models.{config.get("model")}', fromlist=['ClimatePINN']), 'ClimatePINN')
    hidden_dim = config.get('hidden_dim')
    initial_re = config.get('initial_re')
    print(f"Loading model with configuration: hidden_dim={hidden_dim}, initial_re={initial_re}")
    model = ClimatePINN(hidden_dim=hidden_dim, initial_re=initial_re, device=device)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set evaluation mode

    return model, checkpoint['epoch'], config, device

def denormalize_variable(data, var_params):
    """Denormalize variable data using provided min/max parameters."""
    if hasattr(var_params['max'], 'cpu'):
        var_min = var_params['min'].cpu().numpy()
        var_max = var_params['max'].cpu().numpy()
    else:
        var_min = var_params['min']
        var_max = var_params['max']
    return data * (var_max - var_min) + var_min

def generate_predictions(model, dataloader, device, duration=10):
    """Generate predictions for several timesteps."""
    model.eval()
    predictions = []
    targets = []
    norm_params = None

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= duration:
                break
            if norm_params is None:
                norm_params = batch['norm_params']
            inputs = batch['input'].to(device)
            batch_targets = batch['target'].to(device)
            masks = batch['masks'].to(device)
            coords = [coord.to(device) for coord in batch['coords']]

            outputs = model(inputs, masks, coords, compute_physics=False)['output']
            predictions.append(outputs.cpu())
            targets.append(batch_targets.cpu())

            # Explicitly delete tensors and free GPU memory
            del outputs, inputs, batch_targets, masks, coords
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    predictions = torch.cat(predictions, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    return predictions, targets, norm_params

def transform_longitude(arr):
    """Transform longitude by splitting 64 points at index 32 and concatenating."""
    second_half = arr[:, :, :, :32]   
    first_half = arr[:, :, :, 32:]     
    return np.concatenate([first_half, second_half], axis=-1)

# =============================================================================
# Animation Functions
# =============================================================================

def compute_animation_for_scalar(true_data, predicted_data, lat, lon, title, save_path, year, fps=24):
    """Animation for scalar fields (e.g. temperature) comparing true vs predicted."""
    n_timesteps = len(true_data)
    fig = plt.figure(figsize=(20, 12), dpi=300)
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    setup_cartopy_axis(ax1)
    setup_cartopy_axis(ax2)

    # Global color bounds and geographic extent
    vmin = min(np.min(true_data), np.min(predicted_data))
    vmax = max(np.max(true_data), np.max(predicted_data))
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    # Create image artists once with animated=True
    true_img = ax1.imshow(true_data[0], origin='lower', extent=extent,
                           cmap='RdBu_r', vmin=vmin, vmax=vmax, animated=True)
    pred_img = ax2.imshow(predicted_data[0], origin='lower', extent=extent,
                           cmap='RdBu_r', vmin=vmin, vmax=vmax, animated=True)
    # Dynamic titles on each subplot
    title_true = ax1.text(0.5, 1.05, f"True {title} - Time step 0", transform=ax1.transAxes, ha="center")
    title_pred = ax2.text(0.5, 1.05, f"Predicted {title} - Time step 0", transform=ax2.transAxes, ha="center")

    # Single shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(pred_img, cax=cbar_ax)
    cbar.set_label("Temperature (°C)" if "Temperature" in title else title)

    def update_scalar(frame):
        true_img.set_data(true_data[frame])
        pred_img.set_data(predicted_data[frame])
        title_true.set_text(f"True {title} - Time step {frame}")
        title_pred.set_text(f"Predicted {title} - Time step {frame}")
        return [true_img, pred_img, title_true, title_pred]

    writer, save_path = get_animation_writer(save_path, fps, total_frames=n_timesteps)
    fig.suptitle(f'Temperature during year {year}')
    interval = 1000 / fps  # milliseconds per frame
    ani = FuncAnimation(fig, update_scalar, frames=tqdm(range(n_timesteps), leave=False),
                        interval=interval, blit=True)
    print(f"Saving animation to {save_path}")
    ani.save(save_path, writer=writer, dpi=300)
    plt.close(fig)

def compute_animation_for_temperature_difference(true_data, predicted_data, lat, lon, title, save_path, year, fps=24):
    """Animation for the temperature difference between true and predicted fields."""
    n_timesteps = len(true_data)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                           figsize=(12, 8), dpi=300)
    setup_cartopy_axis(ax)

    # Compute global difference bounds and extent
    diff_sample = true_data - predicted_data
    vmin = np.min(diff_sample)
    vmax = np.max(diff_sample)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    diff_data = true_data[0] - predicted_data[0]
    img = ax.imshow(diff_data, origin='lower', extent=extent,
                    cmap='coolwarm', vmin=vmin, vmax=vmax, animated=True)
    title_text = ax.text(0.5, 1.05, "Temperature Difference - Time step 0",
                         transform=ax.transAxes, ha="center")
    cbar = fig.colorbar(img, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label('Temperature Difference (°C)')

    def update_temperature(frame):
        diff = true_data[frame] - predicted_data[frame]
        img.set_data(diff)
        title_text.set_text(f"Temperature Difference - Time step {frame}")
        return [img, title_text]

    writer, save_path = get_animation_writer(save_path, fps, total_frames=n_timesteps)
    fig.suptitle(f'Temperature Difference during year {year}')
    interval = 1000 / fps
    ani = FuncAnimation(fig, update_temperature, frames=tqdm(range(n_timesteps), leave=False),
                        interval=interval, blit=True)
    print(f"Saving animation to {save_path}")
    ani.save(save_path, writer=writer, dpi=300)
    plt.close(fig)

def compute_animation_for_vector_difference(true_vector_data, predicted_vector_data, lat, lon, title, save_path, year, fps=24):
    """Animation for the difference between vector fields (wind) for true vs predicted."""
    n_timesteps = len(true_vector_data)
    lon2d, lat2d = np.meshgrid(lon, lat)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                           figsize=(12, 8), dpi=300)
    setup_cartopy_axis(ax)

    # Downsample for quiver efficiency
    sub = slice(None, None, 2)
    # Initial difference computation for frame 0
    u_true = true_vector_data[0, 0]
    v_true = true_vector_data[0, 1]
    u_pred = predicted_vector_data[0, 0]
    v_pred = predicted_vector_data[0, 1]
    u_diff = u_pred - u_true
    v_diff = v_pred - v_true
    magnitude_diff = np.sqrt(u_diff**2 + v_diff**2)

    q = ax.quiver(lon2d[sub, sub], lat2d[sub, sub],
                  u_diff[sub, sub], v_diff[sub, sub], magnitude_diff[sub, sub],
                  transform=ccrs.PlateCarree(),
                  scale=2,
                  scale_units='xy',
                  cmap='coolwarm',
                  width=0.004,
                  headwidth=4,
                  headlength=5,
                  headaxislength=4.5,
                  minshaft=2)
    title_text = ax.text(0.5, 1.05, f"Difference {title} - Time step 0",
                        transform=ax.transAxes, ha="center")
    cbar = fig.colorbar(q, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label('Wind Speed Difference (m/s)')

    def update_vector(frame):
        u_true = true_vector_data[frame, 0]
        v_true = true_vector_data[frame, 1]
        u_pred = predicted_vector_data[frame, 0]
        v_pred = predicted_vector_data[frame, 1]
        u_diff = u_pred - u_true
        v_diff = v_pred - v_true
        magnitude_diff = np.sqrt(u_diff**2 + v_diff**2)
        q.set_UVC(u_diff[sub, sub], v_diff[sub, sub], magnitude_diff[sub, sub])
        title_text.set_text(f"Difference {title} - Time step {frame}")
        return [q, title_text]

    writer, save_path = get_animation_writer(save_path, fps, total_frames=n_timesteps)
    fig.suptitle(f'Wind Difference during year {year}')
    interval = 1000 / fps
    ani = FuncAnimation(fig, update_vector, frames=tqdm(range(n_timesteps), leave=False),
                        interval=interval, blit=True)
    print(f"Saving animation to {save_path}")
    ani.save(save_path, writer=writer, dpi=300)
    plt.close(fig)

def compute_animation_for_vector(true_vector_data, predicted_vector_data, lat, lon, title, save_path, year, fps=24):
    """Animation for vector fields (wind) comparing true vs predicted."""
    n_timesteps = len(true_vector_data)
    lon2d, lat2d = np.meshgrid(lon, lat)
    fig = plt.figure(figsize=(20, 12), dpi=300)
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    setup_cartopy_axis(ax1)
    setup_cartopy_axis(ax2)

    sub = slice(None, None, 2)
    # Initialize quiver for true wind (frame 0)
    u_true = true_vector_data[0, 0]
    v_true = true_vector_data[0, 1]
    magnitude_true = np.sqrt(u_true**2 + v_true**2)
    q_true = ax1.quiver(lon2d[sub, sub], lat2d[sub, sub],
                        u_true[sub, sub], v_true[sub, sub], magnitude_true[sub, sub],
                        transform=ccrs.PlateCarree(),
                        scale=2,
                        scale_units='xy',
                        cmap='viridis',
                        width=0.004,
                        headwidth=4,
                        headlength=5,
                        headaxislength=4.5,
                        minshaft=2)
    title_true = ax1.text(0.5, 1.05, f"True {title} - Time step 0", transform=ax1.transAxes, ha="center")

    # Initialize quiver for predicted wind (frame 0)
    u_pred = predicted_vector_data[0, 0]
    v_pred = predicted_vector_data[0, 1]
    magnitude_pred = np.sqrt(u_pred**2 + v_pred**2)
    q_pred = ax2.quiver(lon2d[sub, sub], lat2d[sub, sub],
                        u_pred[sub, sub], v_pred[sub, sub], magnitude_pred[sub, sub],
                        transform=ccrs.PlateCarree(),
                        scale=2,
                        scale_units='xy',
                        cmap='viridis',
                        width=0.004,
                        headwidth=4,
                        headlength=5,
                        headaxislength=4.5,
                        minshaft=2)
    title_pred = ax2.text(0.5, 1.05, f"Predicted {title} - Time step 0", transform=ax2.transAxes, ha="center")

    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(q_pred, cax=cbar_ax)
    cbar.set_label('Wind Speed (m/s)')

    def update_vector(frame):
        # Update true wind
        u_true = true_vector_data[frame, 0]
        v_true = true_vector_data[frame, 1]
        magnitude_true = np.sqrt(u_true**2 + v_true**2)
        q_true.set_UVC(u_true[sub, sub], v_true[sub, sub], magnitude_true[sub, sub])
        title_true.set_text(f"True {title} - Time step {frame}")

        # Update predicted wind
        u_pred = predicted_vector_data[frame, 0]
        v_pred = predicted_vector_data[frame, 1]
        magnitude_pred = np.sqrt(u_pred**2 + v_pred**2)
        q_pred.set_UVC(u_pred[sub, sub], v_pred[sub, sub], magnitude_pred[sub, sub])
        title_pred.set_text(f"Predicted {title} - Time step {frame}")

        return [q_true, q_pred, title_true, title_pred]

    writer, save_path = get_animation_writer(save_path, fps, total_frames=n_timesteps)
    fig.suptitle(f'Wind during year {year}')
    interval = 1000 / fps
    ani = FuncAnimation(fig, update_vector, frames=tqdm(range(n_timesteps), leave=False),
                        interval=interval, blit=True)
    print(f"Saving animation to {save_path}")
    ani.save(save_path, writer=writer, dpi=300)
    plt.close(fig)

# =============================================================================
# High-Level Visualization Function
# =============================================================================

def visualize_predictions(run_name, year, fps=24, duration=10, data_dir='./data/era_5_data', save_dir='visualizations'):
    """Generate and save static and animated visualizations."""
    save_dir = os.path.join(save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch, config, device = load_checkpoint(run_name, device=device)
    train_norm_param = load_dataset(
        nb_file=10,
        train_val_split=config.get('train_val_split'),
        root_dir=data_dir,
        normalize=True
    )['train'].get_norm_params()

    dataset_val = ERADataset(
        root_dir=data_dir,
        years=[year],
        normalize=True,
        norm_params=train_norm_param
    )
    val_loader = DataLoader(dataset_val, batch_size=fps, shuffle=False)

    predictions, targets, norm_params = generate_predictions(model, val_loader, device, duration)
    predictions = transform_longitude(predictions)
    targets = transform_longitude(targets)

    # Geographic coordinates
    lat = np.linspace(-90, 90, predictions.shape[-2])
    lon = np.linspace(-180, 180, predictions.shape[-1])

    # Process temperature predictions
    temp_pred = predictions[:, 0]
    temp_true = targets[:, 0]
    temp_pred = denormalize_variable(temp_pred, norm_params['2m_temperature'])
    temp_true = denormalize_variable(temp_true, norm_params['2m_temperature'])
    temp_pred = temp_pred - 273.15  # Kelvin -> Celsius
    temp_true = temp_true - 273.15

    compute_animation_for_scalar(temp_true, temp_pred, lat, lon, "Temperature (°C)",
        os.path.join(save_dir, f'temperature_prediction_{year}.mp4'), year, fps=fps)

    compute_animation_for_temperature_difference(temp_true, temp_pred, lat, lon, "Temperature (°C)", 
        os.path.join(save_dir, f'temperature_prediction_{year}_comp.mp4'), year, fps=fps)

    # Process wind predictions
    wind_pred = predictions[:, 1:3]
    wind_true = targets[:, 1:3]
    wind_pred[:, 0] = denormalize_variable(wind_pred[:, 0], norm_params['10m_u_component_of_wind'])
    wind_true[:, 0] = denormalize_variable(wind_true[:, 0], norm_params['10m_u_component_of_wind'])
    wind_pred[:, 1] = denormalize_variable(wind_pred[:, 1], norm_params['10m_v_component_of_wind'])
    wind_true[:, 1] = denormalize_variable(wind_true[:, 1], norm_params['10m_v_component_of_wind'])

    compute_animation_for_vector(wind_true, wind_pred, lat, lon, "Predicted Wind (m/s)", 
        os.path.join(save_dir, f'wind_prediction_{year}.mp4'), year, fps=fps)

    compute_animation_for_vector_difference(wind_true, wind_pred, lat, lon, "Predicted Wind (m/s)", 
        os.path.join(save_dir, f'wind_prediction_{year}_diff.mp4'), year, fps=fps)

if __name__ == "__main__":
    runs = ['run_4', 'run_7', 'run_8', 'run_9']
    fps = 5
    year = 2000
    duration = 10

    for run in tqdm(runs):
        visualize_predictions('run_8', year, fps=fps, duration=duration)
        break
