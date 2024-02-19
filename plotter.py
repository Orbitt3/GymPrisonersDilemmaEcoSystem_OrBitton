import datetime
import numpy as np
import tensorflow as tf
import io
import matplotlib.pyplot as plt


class Plotter:
  @staticmethod
  def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory.

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

  @staticmethod
  def image_grid(imaged_plots):
    """
    Return a 5x5 grid of the images as a matplotlib figure.
    imaged_plots should be a dict containing images and topics for each image
    """
    # Create a figure to contain the plot.
    # imaged_plots = {'class_name': ['q0', 'q1', 'q1-q0'], 'images': imaged_plots}

    images_names = imaged_plots['class_name']
    images = imaged_plots['images']
    axes_size = len(images_names)

    figure = plt.figure(figsize=(15, 15))
    fig, ax = plt.subplots(axes_size, figsize=(10, 10))

    for i in range(axes_size):
      # Start next subplot.

      im = ax[i].imshow(images[i], cmap="Wistia")
      cbar = ax[i].figure.colorbar(im, ax=ax[i])

      # Rotate the tick labels to be more legible
      plt.setp(ax[i].get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
      ax[i].set_title(f"{images_names[i]} Table", size=20)
      fig.tight_layout()

    return fig





  @staticmethod
  def image_grid_beta(imaged_plots): # not in use
    """
    Return a 5x5 grid of the images as a matplotlib figure.
    imaged_plots should be a dict containing images and topics for each image
    """
    # Create a figure to contain the plot.
    imaged_plots = {'class_name': ['q0', 'q1', 'q1-q0'], 'images': imaged_plots}

    images_names = imaged_plots['class_name']
    images = imaged_plots['images']

    figure = plt.figure(figsize=(15, 15))

    for i in range(3):
      # Start next subplot.

      plt.subplot(2, 2, i + 1, title=images_names[i])
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)

      plt.imshow(images[i], cmap='Wistia')
    return figure
