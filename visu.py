import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl

# Enable anti-aliasing globally
mpl.rcParams['text.antialiased'] = True
mpl.rcParams['lines.antialiased'] = True
mpl.use('Agg')

cbar = None

ABSOLUTE_ZERO = -273.15  # in degrees celsius


def kelvin_to_celsius(temp):
    return temp + ABSOLUTE_ZERO


def compute_animation_for_vectors(vectors, save_path, titles=None, cbar_label=None, lat_ext=[-90, 90], lon_ext=[-180, 180]):
    """

    :param vectors: (n_scalars, time, dimensions(lat and lon), lat, lon)
    :param lat_ext:
    :param lon_ext:
    :param save_path:
    :return:
    """
    n_subplots = len(vectors)
    n_timesteps = vectors[0].shape[0]
    colormap = 'viridis'

    H, W = vectors[0].shape[-2:]
    lat = np.linspace(lat_ext[0], lat_ext[1], H)  # H points from -90 to 90 (latitude)
    lon = np.linspace(lon_ext[0], lon_ext[1], W)  # W points from -180 to 180 (longitude)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Set up the figure and axis
    fig, axs = plt.subplots(nrows=n_subplots, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 8), dpi=300)

    if n_subplots == 1:
        axs = [axs]

    # Compute the wind magnitude
    magnitudes = [np.sqrt(vectors[i][:, 0] ** 2 + vectors[i][:, 1] ** 2) for i in range(n_subplots)]

    # To have a consistent colorbar over all the busplots
    vmin = min([s_.min() for s_ in magnitudes])
    vmax = min([s_.max() for s_ in magnitudes])
    normalizer = plt.Normalize(vmin, vmax)
    cbar_mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=colormap)
    cbar = fig.colorbar(cbar_mappable, ax=axs)
    cbar.set_label(cbar_label)

    def update_vectors(frame):

        for i, ax in enumerate(axs):
            ax.clear()
            ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
            ax.coastlines()
            ax.set_title(titles[i])  # Customize title as needed

            # Plot the wind vectors with color based on magnitude
            ax.quiver(
                lon2d, lat2d,
                vectors[i][frame, 0], vectors[i][frame, 1],
                magnitudes[i][frame],
                transform=ccrs.PlateCarree(),
                # scale=1,
                scale_units='xy',
                width=0.0018,
                cmap=colormap
            )

    # Create an animation
    ani = FuncAnimation(fig, update_vectors, frames=n_timesteps, interval=200, blit=False)

    # Save the animation with high quality
    ani.save(save_path, writer="ffmpeg", fps=1, dpi=300, bitrate=5000)
    cbar = None
    plt.close()


def compute_animation_for_scalars(scalars, save_path, titles=None, cbar_label=None, lat_ext=[-90, 90], lon_ext=[-180, 180]):
    """

    :param scalars: (n_scalars, time, lat, lon)
    :param lat_ext:
    :param lon_ext:
    :param save_path:
    :return:
    """
    n_scalars = len(scalars)
    n_timesteps = scalars[0].shape[0]
    colormap = 'coolwarm'

    extent = lon_ext + lat_ext

    # Set up the figure and axis
    fig, axs = plt.subplots(nrows=n_scalars, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 8), dpi=300)

    if n_scalars == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
        ax.coastlines()
        ax.set_title(titles[i])  # Customize title as needed

    # To have a consistent colorbar over all the busplots
    vmin = min([s_.min() for s_ in scalars])
    vmax = min([s_.max() for s_ in scalars])
    normalizer = plt.Normalize(vmin, vmax)
    cbar_mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=colormap)
    cbar = fig.colorbar(cbar_mappable, ax=axs)
    cbar.set_label(cbar_label)

    def update_scalars(frame):
        # ax.clear()
        for i, ax in enumerate(axs):
            ax.imshow(scalars[i][frame], origin='lower', extent=extent, cmap=colormap, norm=normalizer)

    # Create an animation
    ani = FuncAnimation(fig, update_scalars, frames=n_timesteps, interval=200, blit=False)

    # Save the animation with high quality
    ani.save(save_path, writer="ffmpeg", fps=1, dpi=300, bitrate=5000)
    plt.close()


