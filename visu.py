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
from models import ConditionalFlowGenerator2d
# Render settings
mpl.rcParams['text.antialiased'] = True
mpl.rcParams['lines.antialiased'] = True
mpl.use('Agg')
import argparse
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

def load_checkpoint_cf(checkpoint_path, device):
    """
    Charge le checkpoint pour le générateur conditionnel.
    On récupère le nombre de flows (par défaut 4 si non précisé dans le checkpoint).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    nb_flows = checkpoint.get("nb_flows", 16)
    gen = ConditionalFlowGenerator2d(
        context_channels=7,
        latent_channels=3,
        num_flows=nb_flows
    ).to(device)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen.eval()
    return gen, checkpoint

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
    sub = slice(None, None, 1)
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

    sub = slice(None, None, 1)
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

def visualize_predictions_cf(checkpoint_path, year, fps, duration, data_dir, save_dir):
    """
    Charge le modèle à partir d'un checkpoint et génère une vidéo finale.
    La méthode sample_most_probable (ici utilisée comme sample_max_prob) est appelée avec num_samples=100.
    """
    # Charger le dataset de validation
    datasets = load_dataset(
        nb_file=10,
        train_val_split=0.8,
        year0=1979,
        root_dir=data_dir,
        normalize=True
    )
    val_dataset = datasets["val"]
    val_loader = DataLoader(val_dataset, batch_size=fps, shuffle=False, num_workers=5)

    # Récupérer un batch de données pour générer la vidéo
    batch_data = next(iter(val_loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = batch_data["input"].to(device)
    masks = batch_data["masks"].to(device)
    lat_coord = batch_data["coords"][0].unsqueeze(1).to(device)
    lon_coord = batch_data["coords"][1].unsqueeze(1).to(device)
    coords = torch.cat([lat_coord, lon_coord], dim=1)
    x = torch.cat([inputs, masks, coords], dim=1)
    # On remet dans l'ordre attendu par le modèle (B, channels, width, height)
    x = x.permute(0, 3, 2, 1)
    
    # Charger le générateur depuis le checkpoint
    gen, _ = load_checkpoint_cf(checkpoint_path, device)
    
    # Générer la prédiction en utilisant 100 échantillons (méthode sample_max_prob)
    # Ici, on considère que sample_max_prob est équivalent à sample_most_probable
    fake = gen.sample_most_probable(x, num_samples=100)
    
    # Réorganiser les dimensions pour la visualisation : (num_samples, width, height, channels)
    fake = fake.permute(0, 3, 2, 1).cpu().detach().numpy()
    # On transforme les longitudes si besoin (fonction déjà présente)
    fake = transform_longitude(fake)
    
    # Récupérer les paramètres de normalisation du dataset
    norm_params = val_dataset.get_norm_params()
    
    # Traitement pour la température (canal 0)
    temp_pred = fake[:, :, :, 0]
    if val_dataset.normalize:
        temp_pred = denormalize_variable(temp_pred, norm_params['2m_temperature']) - 273.15
    else:
        temp_pred = temp_pred - 273.15

    # Définir la grille géographique
    nlat = temp_pred.shape[1]
    nlon = temp_pred.shape[2]
    lat_vals = np.linspace(-90, 90, nlat)
    lon_vals = np.linspace(-180, 180, nlon)
    
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, f"final_temperature_video_{year}.mp4")
    
    # Générer la vidéo (ici, on affiche la température prédite sur chaque frame)
    compute_animation_for_scalar(
         true_data=temp_pred,         # Pour la vidéo finale, true_data et predicted_data sont identiques
         predicted_data=temp_pred,
         lat=lat_vals,
         lon=lon_vals,
         title="Final Temperature (°C)",
         save_path=video_path,
         year=year,
         fps=fps
    )
    print(f"Vidéo finale sauvegardée ici : {video_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Génération de la vidéo finale pour le ConditionalFlowGenerator via un checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Chemin vers le fichier de checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, default="/home/ensta/ensta-cesar/era_5_data/", help="Répertoire contenant les données")
    parser.add_argument("--year", type=int, default=2000, help="Année à visualiser")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second pour la vidéo")
    parser.add_argument("--duration", type=int, default=10, help="Nombre de timesteps (durée) pour la vidéo")
    parser.add_argument("--save_dir", type=str, default="visualizations", help="Répertoire de sauvegarde de la vidéo finale")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil : {device}")
    
    visualize_predictions_cf(
        checkpoint_path=args.checkpoint,
        year=args.year,
        fps=args.fps,
        duration=args.duration,
        data_dir=args.data_dir,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()