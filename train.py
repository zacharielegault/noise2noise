import time
import os
import argparse
import numpy as np

from keras.callbacks import TerminateOnNaN, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam

from models import unet
from utils import get_psnr, Logger, NoiseGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training script")

    # Required arguments
    parser.add_argument(
        "-d", "--data_directory",
        required=True,
        help="Directory containing the dataset as numpy archives (files " \
             "should be called `<data_directory>/train.npy` and " \
             "`<data_directory>/valid.npy`).")
    parser.add_argument(
        "-l", "--log_directory",
        required=True,
        help="Directory to save the logs and model checkpoints.")
    parser.add_argument(
        "-s", "--sigma",
        required=True, type=float,
        help="Noise level on the data.")

    # Optional arguments
    parser.add_argument(
        "-n", "--noise2noise",
        action='store_true',
        help="Optional. Boolean. Use Noise2Noise algorithm.")
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=32,
        help="Optional. Batch size. Default is 32")
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=1,
        help="Optional. Number of epochs. Default is 1.")
    parser.add_argument(
        "-v", "--verbose",
        type=int,
        default=1,
        help="Optional. Verbosity level of Keras. Default is 1.")

    return parser.parse_args()


def main():
    args = parse_args()

    timestr = time.strftime("%Y%m%d%H%M%S")

    log_dir_parent = args.log_directory
    log_dir = log_dir_parent + "/{}".format(timestr)
    checkpoints_dir = log_dir + "/checkpoints"
    os.mkdir(log_dir)
    os.mkdir(checkpoints_dir)

    optimizer = Adam()

    train = np.load(args.data_directory + "/train.npy")
    valid = np.load(args.data_directory + "/valid.npy")

    callbacks = [
        Logger(
            filename=log_dir + "/{}.log".format(timestr),
            optimizer=optimizer,
            sigma=args.sigma,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dataset_dir=args.data_directory,
            checkpoints_dir=checkpoints_dir,
            noise2noise=args.noise2noise,
            dtype=train.dtype),
        CSVLogger(log_dir + "/{}.csv".format(timestr)),
        TerminateOnNaN(),
        ModelCheckpoint(
            checkpoints_dir + '/checkpoint.' + timestr + \
            '.{epoch:03d}-{val_loss:.3f}-{val_psnr:.5f}.h5',
            monitor='val_psnr',
            mode='max',
            save_best_only=True)
    ]

    # Build PSNR function
    if train.dtype == np.uint8:
        psnr = get_psnr(8)
    elif train.dtype == np.uint16:
        psnr = get_psnr(16)
    elif np.issubdtype(train.dtype, np.floating):
        psnr = get_psnr(1)
    else:
        raise TypeError

    model = unet(shape=(None, None, 1))
    model.compile(optimizer=optimizer, loss='mse', metrics=[psnr])

    # Same noise for all training procedure
    # train_input = NoiseGenerator.add_gaussian_noise(train, args.sigma)
    # if args.noise2noise:
    #     train_target = NoiseGenerator.add_gaussian_noise(train, args.sigma)
    # else:
    #     train_target = train

    # valid_input = NoiseGenerator.add_gaussian_noise(valid, args.sigma)
    # valid_target = valid  # Validation images should be clean

    # hist = model.fit(train_input, train_target,
    #                  epochs=args.epochs,
    #                  batch_size=args.batch_size,
    #                  verbose=args.verbose,
    #                  callbacks=callbacks,
    #                  validation_data=(valid_input, valid_target))

    # New noise for each new batch
    train_generator = NoiseGenerator(data=train,
                                     batch_size=args.batch_size,
                                     sigma=args.sigma,
                                     noise2noise=args.noise2noise)
    valid_generator = NoiseGenerator(data=valid,
                                     batch_size=args.batch_size,
                                     sigma=args.sigma,
                                     noise2noise=False)
    
    model.fit_generator(generator=train_generator,
                        epochs=args.epochs,
                        verbose=args.verbose,
                        callbacks=callbacks,
                        validation_data=valid_generator)


if __name__ == '__main__':
    main()
