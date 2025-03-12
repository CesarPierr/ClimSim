import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import wandb
from tqdm import tqdm   # <--- for progress bar

mpl.rcParams['text.antialiased'] = True
mpl.rcParams['lines.antialiased'] = True
mpl.use('Agg')

# A custom writer class that adds a tqdm progress bar while frames are being written.
class TQDMWriter(FFMpegWriter):
    def __init__(self, total_frames, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pbar = tqdm(total=total_frames, desc="Rendering frames", unit="frame")

    def grab_frame(self, *args, **kwargs):
        super().grab_frame(*args, **kwargs)
        self._pbar.update(1)

    def finish(self):
        super().finish()
        self._pbar.close()


def compute_animation_for_scalars(
    scalars,
    save_path,
    titles=None,
    cbar_label=None,
    fps=1,
    lat_ext=[-90, 90],
    lon_ext=[0, 360],
    log_wandb=False
):
    """
    Faster animation for scalar fields (e.g. temperature) with Cartopy, 
    plus a progress bar for the rendering process.

    Parameters
    ----------
    scalars : list/array of shape (n_subplots, T, lat, lon)
        For each subplot, an array of shape (T, lat, lon).
    save_path : str
        Where to save the resulting .mp4 video file.
    titles : list of str, optional
        Titles for each subplot.
    cbar_label : str, optional
        Label for the colorbar.
    fps : int
        Frames per second for the output video.
    lat_ext : list [min_lat, max_lat]
        Latitude extent.
    lon_ext : list [min_lon, max_lon]
        Longitude extent.
    log_wandb : bool
        If True, log the resulting .mp4 as a wandb.Video.
    """
    # Number of subplots
    n_subplots = len(scalars)
    # Number of frames (timesteps)
    n_timesteps = scalars[0].shape[0]

    colormap = 'coolwarm'
    extent = lon_ext + lat_ext

    fig, axs = plt.subplots(
        nrows=n_subplots,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        figsize=(12, 8),
        dpi=100
    )

    if n_subplots == 1:
        axs = [axs]

    # Draw static features once (land, ocean, coastlines, etc.)
    for i, ax in enumerate(axs):
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
        ax.coastlines()
        if titles is not None:
            ax.set_title(titles[i])

    # Determine global min/max for color normalization
    vmin = min([s_.min() for s_ in scalars])
    vmax = max([s_.max() for s_ in scalars])
    normalizer = plt.Normalize(vmin=vmin, vmax=vmax)

    # Create the colorbar once for all subplots
    cbar_mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=colormap)
    cbar = fig.colorbar(cbar_mappable, ax=axs, orientation='vertical')
    if cbar_label:
        cbar.set_label(cbar_label)

    # Create image objects (one per subplot) only once
    images = []
    for i, ax in enumerate(axs):
        # Initial frame = 0
        im = ax.imshow(
            scalars[i][0],
            origin='lower',
            extent=extent,
            cmap=colormap,
            norm=normalizer,
            transform=ccrs.PlateCarree(central_longitude=180)
        )
        images.append(im)

    def init_func():
        # Return the list of artists to be re-drawn
        return images

    def update(frame):
        for i in range(n_subplots):
            images[i].set_data(scalars[i][frame])
        return images

    ani = FuncAnimation(
        fig,
        update,
        frames=n_timesteps,
        init_func=init_func,
        interval=200,
        blit=False
    )

    # Use our custom TQDMWriter with a progress bar
    writer = TQDMWriter(total_frames=n_timesteps, fps=fps, bitrate=3000)

    # Save the animation
    ani.save(save_path, writer=writer, dpi=100)

    if log_wandb:
        video = wandb.Video(save_path, fps=fps, format="mp4")
        wandb.log({"scalar_animation": video})

    plt.close(fig)


def compute_animation_for_vectors(
    vectors,
    save_path,
    titles=None,
    cbar_label=None,
    fps=1,
    lat_ext=[-90, 90],
    lon_ext=[0, 360],
    log_wandb=False
):
    """
    Faster animation for vector fields (e.g. wind) with Cartopy, plus a progress bar.

    Parameters
    ----------
    vectors : list/array of shape (n_subplots, T, 2, lat, lon)
        For each subplot, an array representing U, V at each timestep:
        vectors[i][t, 0, :, :] = U(t)
        vectors[i][t, 1, :, :] = V(t)
    save_path : str
        Where to save the resulting .mp4 video file.
    titles : list of str, optional
        Titles for each subplot.
    cbar_label : str, optional
        Label for the colorbar (e.g., "Wind speed").
    fps : int
        Frames per second for the output video.
    lat_ext : list [min_lat, max_lat]
        Latitude extent.
    lon_ext : list [min_lon, max_lon]
        Longitude extent.
    log_wandb : bool
        If True, log the resulting .mp4 as a wandb.Video.
    """
    n_subplots = len(vectors)
    n_timesteps = vectors[0].shape[0]

    H, W = vectors[0].shape[-2], vectors[0].shape[-1]
    lat = np.linspace(lat_ext[0], lat_ext[1], H)
    lon = np.linspace(lon_ext[0], lon_ext[1], W)
    lon2d, lat2d = np.meshgrid(lon, lat)

    colormap = 'viridis'

    # Calculate magnitudes for coloring
    magnitudes = [np.sqrt(vectors[i][:, 0]**2 + vectors[i][:, 1]**2) for i in range(n_subplots)]
    mag_min = min([m.min() for m in magnitudes])
    mag_max = max([m.max() for m in magnitudes])
    normalizer = plt.Normalize(vmin=mag_min, vmax=mag_max)

    fig, axs = plt.subplots(
        nrows=n_subplots,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        figsize=(12, 8),
        dpi=100
    )

    if n_subplots == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
        ax.coastlines()
        if titles is not None:
            ax.set_title(titles[i])

    cbar_mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=colormap)
    cbar = fig.colorbar(cbar_mappable, ax=axs, orientation='vertical')
    if cbar_label:
        cbar.set_label(cbar_label)

    quiver_plots = []
    for i, ax in enumerate(axs):
        U0 = vectors[i][0, 0]
        V0 = vectors[i][0, 1]
        M0 = magnitudes[i][0]

        quiv = ax.quiver(
            lon2d,
            lat2d,
            U0,
            V0,
            M0,
            transform=ccrs.PlateCarree(central_longitude=180),
            scale_units='xy',
            cmap=colormap,
            norm=normalizer,
            width=0.0018
        )
        quiver_plots.append(quiv)

    def init_func():
        return quiver_plots

    def update(frame):
        for i in range(n_subplots):
            U = vectors[i][frame, 0]
            V = vectors[i][frame, 1]
            M = magnitudes[i][frame]
            quiver_plots[i].set_UVC(U, V, M)
        return quiver_plots

    ani = FuncAnimation(
        fig,
        update,
        frames=n_timesteps,
        init_func=init_func,
        interval=200,
        blit=False
    )

    writer = TQDMWriter(total_frames=n_timesteps, fps=fps, bitrate=3000)
    ani.save(save_path, writer=writer, dpi=100)

    if log_wandb:
        video = wandb.Video(save_path, fps=fps, format="mp4")
        wandb.log({"vector_animation": video})

    plt.close(fig)
