from io import IOBase

from math import ceil
import numpy as np

from skimage.util import view_as_windows

import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras.utils.data_utils import Sequence

from tensorflow.image import total_variation as tv


def get_psnr(bit_depth):
    """Build PSNR function depending on bit-depth.

    Args:
        bit_depth: integer or float. Bit depth of the images. If using
            floating point data, bit depth should normally be 1.
    """
    max_val = 2**bit_depth - 1

    def psnr(y_true, y_pred):
        """Compute image PSNR in dB.

        Args:
            y_true: tensor of true targets.
            y_pred: tensor of predicted targets.
        """
        return (10.0 * K.log(max_val**2 / (K.mean(K.square(y_pred - y_true))))
                / K.log(10.0))

    return psnr


def total_variation(y_true, y_pred):
    """Compute scalar value of total variation loss.

    Args:
        y_true: tensor of true targets. Argument is not used, but still
            there to match the expected format.
        y_pred: tensor of predicted targets.
    """
    return K.sum(tv(y_pred))


class Logger(Callback):
    """Additionnal custom class to log experiment details. Extends the
    `keras.callbacks.Callback` base class.

    Note that this class does NOT replace the other Keras callbacks
    (`CSVLogger`, `ModelCheckpoint`, etc.). Those should be added
    separately to the model's callback list.

    Args:
        filename: string. Complete filename where to write the log
            file.
        optimizer: keras.optimizers.Optimizer. Optimizer object.
        sigma: float. Noise level, typically the standard deviation of
            additive Gaussian noise.
        batch_size: integer. Batch size used during training.
        epochs: integer. Number of epochs.
        dataset_dir: string. Directory where dataset is stored.
        checkpoints_dir: string. Directory where to store model
            checkpoints.
        noise2noise: boolean. Whether the experiment uses
            Noise2Noise or Noise2Clean.
        dtype: type. Data type of the images.
        total_variation_weigth: float. Weight given to secondary
            total variation loss, if using.
    """

    def __init__(self,
                 filename: str,
                 optimizer: Optimizer,
                 sigma: float,
                 batch_size: int,
                 epochs: int,
                 dataset_dir: str,
                 checkpoints_dir: str,
                 noise2noise: bool,
                 dtype: type,
                 total_variation_weigth=0):
        super(Logger, self).__init__()
        self.filename = filename
        self.optimizer = optimizer
        self.sigma = sigma
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset_dir = dataset_dir
        self.checkpoints_dir = checkpoints_dir
        self.noise2noise = noise2noise
        self.dtype = dtype
        self.total_variation_weigth = total_variation_weigth

    def on_train_begin(self, logs=None):
        with open(self.filename, "w") as f:
            self._log_model(f)
            self._log_optimizer(f)
            self._log_fit_params(f)
            self._log_tv(f)
            self._log_directories(f)
            self._log_noise(f)

    def on_train_end(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def _log_tv(self, f: IOBase):
        f.write("Total variation weigth: {}\n".format(self.total_variation_weigth))
        f.write("_________________________________________________________________\n\n")

    def _log_directories(self, f: IOBase):
        f.write("Dataset directory: {}\n".format(self.dataset_dir))
        f.write("dtype: {}\n".format(self.dtype))
        f.write("Checkpoints directory: {}\n".format(self.checkpoints_dir))
        f.write("_________________________________________________________________\n\n")

    def _log_fit_params(self, f: IOBase):
        f.write("Fit parameters\n")
        f.write("Epochs: {}\n".format(self.epochs))
        f.write("Batch size: {}\n".format(self.batch_size))
        f.write("_________________________________________________________________\n\n")

    def _log_model(self, f: IOBase):
        f.write("Model: {}\n".format(self.model.name))
        self.model.summary(line_length=120, print_fn=lambda x: f.write(x + '\n'))
        f.write("\n")

    def _log_optimizer(self, f: IOBase):
        opt_name = self.optimizer.__class__.__name__
        f.writelines("Optimizer: {}\n".format(opt_name))
        if opt_name == "Adam":
            f.writelines(["lr: {}\n".format(K.eval(self.optimizer.lr)),
                          "beta_1: {}\n".format(K.eval(self.optimizer.beta_1)),
                          "beta_2: {}\n".format(K.eval(self.optimizer.beta_2)),
                          "epsilon: {}\n".format(self.optimizer.epsilon),
                          "decay: {}\n".format(K.eval(self.optimizer.decay))])
        elif opt_name == "SGD":
            f.writelines(["lr: {}\n".format(K.eval(self.optimizer.lr)),
                          "momentum: {}\n".format(K.eval(self.optimizer.momentum)),
                          "decay: {}\n".format(K.eval(self.optimizer.decay)),
                          "initial_decay: {}\n".format(self.optimizer.initial_decay),
                          "nesterov: {}\n".format(self.optimizer.nesterov)])

        f.write("_________________________________________________________________\n\n")

    def _log_noise(self, f: IOBase):
        if isinstance(self.sigma, dict):
            # Attribute sigma should not be a dict unless using the FuelCell dataset
            assert list(self.sigma.keys()) == ['A', 'B', 'C', 'D', 'E']

        f.write("Noise2Noise: {}\n".format(self.noise2noise))
        f.write("Noise level: {}\n".format(self.sigma))
        f.write("_________________________________________________________________\n\n")


class NoiseGenerator(Sequence):
    """A subclass of keras.utils.data_utils.Sequence. Yields batches
    of (input, target) tuples from the clean images. The input is a
    batch of noisy images, and the target is the batch of the same
    images, either clean or noisy (different from the noise added to
    the input).

    Args:
        data: numpy array. The data which is fed to the training (or
            validation) procedure.
        batch_size: integer.
        sigma: float. Standard deviation of the Gaussian noise
        noise2noise: boolean. Whether or not noise is added to the
            target.
    """
    
    def __init__(self,
                 data: np.ndarray,
                 batch_size: int,
                 sigma: float,
                 noise2noise: bool):
        self.data = data
        self.batch_size = batch_size
        self.sigma = sigma
        self.noise2noise = noise2noise

    def __len__(self):
        return abs(int(np.ceil(len(self.data) / float(self.batch_size))))

    def __getitem__(self, idx):
        batch_x = self.add_gaussian_noise(
            self.data[idx * self.batch_size:(idx + 1) * self.batch_size],
            self.sigma)

        if self.noise2noise:
            batch_y = self.add_gaussian_noise(
                self.data[idx * self.batch_size:(idx + 1) * self.batch_size],
                self.sigma)
        else:
            batch_y = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y

    @classmethod
    def add_gaussian_noise(cls, image: np.ndarray, sigma: float):
        """Add zero-mean Gaussian white noise to images. Clips the values
        between 0 and the max of the data type for integer data, and
        between 0 and 1 floating point data.

        Args:
            image: np.ndarray. Original images to which noise is added.
            sigma: float. Standard deviation of the Gaussian.
        """
        noise = np.random.normal(scale=sigma, size=image.shape)

        noisy_image = image.astype(np.float)
        noisy_image += noise

        if image.dtype == np.uint8:
            clip = 2**8 - 1
        elif image.dtype == np.uint16:
            clip = 2**16 - 1
        elif np.issubdtype(image.dtype, np.floating):
            clip = 1
        else:
            raise TypeError

        noisy_image = np.clip(noisy_image, 0, clip).astype(image.dtype)

        return noisy_image


def get_patches(images: np.ndarray, patch_size: int, patch_overlap: float):
    """Cut images into square patches. If the patch size is not a
    divider of the image size, the image is reflected to complete
    patches on the border.

    Args:
        images: numpy array. The array is assumed to be 4
            dimensions (NHWC).
        patch_size: integer.
        patch_overlap: float between 0 and 1. Ratio of how much 2
            neighbooring patches share pixel values. A value of 0 gives
            completely disjoint subsets of the images.
    """
    if images.shape[1] % patch_size != 0:
        pad_h = patch_size - images.shape[1]%patch_size
    else:
        pad_h = 0

    if images.shape[2] % patch_size != 0:
        pad_w = patch_size - images.shape[2]%patch_size
    else:
        pad_w = 0

    pad = [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
    images = np.pad(images, pad_width=pad, mode="reflect")

    window_shape = (images.shape[0], patch_size, patch_size, images.shape[3])
    step = (images.shape[0],
            ceil(patch_size * (1 - patch_overlap)),
            ceil(patch_size * (1 - patch_overlap)),
            images.shape[3])
    images = view_as_windows(images, window_shape=window_shape, step=step)

    s = images.shape
    new_shape = (s[0]*s[1]*s[2]*s[3]*s[4],
                 s[5],
                 s[6],
                 s[7])

    images = np.reshape(images, new_shape)

    return images
