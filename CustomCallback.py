import os
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib import cm
import numpy as np
from PIL import ImageDraw, ImageFont, Image


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0, log_path="", env=None):
        super(CustomCallback, self).__init__(verbose)
        self.env = env.envs[0]
        self.log_path = log_path
        self.imgs = []
        self.iterations = 1
        self.plot_freq = 8

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self):
        """
        This method will be called by the callback at each step.
        """
        self.iterations += 1
        if self.iterations % self.plot_freq == 0:
            # Access the data
            obs = self.env.low_env.score.astype(float)

            # Define cell size
            cell_size = 40  # Increase cell size for a bigger image
            img_size = obs.shape[0] * cell_size

            # Prepare a blank image
            img = Image.new('RGB', (img_size, img_size), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            font = ImageFont.load_default()

            for i in range(obs.shape[0]):
                for j in range(obs.shape[1]):
                    cell_value = obs[i, j]
                    # Define the position for each cell value (adjust as needed)
                    position = (j * cell_size, i * cell_size)
                    # Draw the cell value
                    d.text(position, str(cell_value), font=font, fill=(255, 255, 255))

            self.imgs.append(img)  # Append the image to the list

    def _on_step2(self) -> bool:
        self.iterations += 1
        data = self.env.low_env.score.astype(float)

        # normalize data to global min and max
        data = (data - (-10)) / (10 - (-10))  # considering all possible values from -10 to 10


        # apply colormap
        colored_data = cm.viridis(data)

        # convert to 8-bit RGB
        image_data = (255 * colored_data).astype(np.uint8)
        # create PIL image
        image = Image.fromarray(image_data)
        # print('image_data', image_data) # 2x4x3 image

        # create ImageDraw object
        draw = ImageDraw.Draw(image)

        # add text to image
        text = str(self.iterations)
        font = ImageFont.load_default()
        draw.text((10, 10), text, font=font, fill=(0, 0, 0))  # black text in upper-left corner

        # resize image
        resized_image = image.resize((image.size[0] * 250, image.size[1] * 250))

        self.imgs.append(resized_image)
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        f_name = os.path.join(self.logger.dir, f"iteration_{self.iterations}")
        # resized_array = np.repeat(np.repeat(data, 9, axis=0), 9, axis=1)

        # resized_imgs = [np.repeat(np.repeat(img, 9, axis=0), 9, axis=1) for img in
        #                 self.imgs]  # Double the size of all images
        self.imgs[0].save(f"gif.gif", save_all=True, append_images=self.imgs[3::4],
                          duration=200)  # 200ms between frames
        pass