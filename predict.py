import argparse

import numpy as np
from skimage.io import imread, imsave

import keras
import keras.backend as K
import matplotlib.pyplot as plt

from utils import get_psnr, NoiseGenerator, get_patches


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize the output of the model.")

    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the model.")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input file. Can be a '.npy', '.png', '.jpg', '.jpeg' or '.tif'.")
    parser.add_argument(
        "-s", "--sigma",
        required=True,
        type=float,
        help="Standard deviation of the gaussian noise.")

    parser.add_argument(
        "-p", "--patch",
        default=256,
        type=int,
        help="Optional. Patch size. Default is 256.")
    parser.add_argument(
        "-o", "--output",
        default="",
        help="Optional. Output file. A '.npz' file will save reshaped input " \
             "image, noisy image and prediction. A '.npy' will save the " \
             "prediction array. A '.png', '.jpg', '.jpeg' or '.tif' will " \
             "save the prediction, but only if the input is a single image")
    parser.add_argument(
        "-v", "--visualize",
        default=False,
        action='store_true',
        help="Optional. Boolean. Visualize the prediction.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.input.lower().endswith(".npy"):
        image = np.load(args.input)
    elif args.input.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
        image = imread(args.input)
    else:
        raise ValueError
    
    print("Read image {} with shape {}.".format(args.input, image.shape))

    if image.dtype == np.uint8:
        bit_depth = 8
    elif image.dtype == np.uint16:
        bit_depth = 16
    elif np.issubdtype(image.dtype, np.floating):
        bit_depth = 1
    else:
        raise TypeError

    psnr = get_psnr(bit_depth)

    model = keras.models.load_model(args.model, custom_objects={"psnr": psnr})

    # Transform into 4D tensor
    if image.ndim == 2:
        image = image[np.newaxis, :, :, np.newaxis]
    elif image.ndim == 3:
        image = image[np.newaxis, :, :]
    elif image.ndim != 4:
        raise Exception("Error with image dimensions: ndim should be 2 or " \
            "3, but received {}".format(image.ndim))

    patch_size = args.patch
    print("Cutting images into {}x{} patches.".format(patch_size, patch_size))
    image = get_patches(image, patch_size, 0)

    print("Adding noise with standard deviation {}.".format(args.sigma))
    noisy = NoiseGenerator.add_gaussian_noise(image, args.sigma)

    print("Making prediction on input tensor with shape {}".format(noisy.shape))
    pred = model.predict(noisy, batch_size=1)

    pred_psnr = psnr(image, pred)
    print("PSNR of prediction: {}".format(K.eval(pred_psnr)))

    if args.output:
        if args.output.lower().endswith(".npz"):
            print("Saving image, noisy image and prediction to {}".format(
                args.output))
            np.savez(args.output, image=image, noisy=noisy, pred=pred)
        elif args.output.lower().endswith(".npy"):
            print("Saving prediction to {}".format(args.output))
            np.save(args.output, pred)
        elif args.output.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            print("Saving prediction to {}".format(args.output))
            for i, p in enumerate(pred):
                extension_dot_idx = args.output.rfind(".")
                output = args.output[:extension_dot_idx] + "_" + str(i + 1) \
                    + args.output[extension_dot_idx:]
                imsave(output, np.clip(np.squeeze(p), 0, bit_depth). \
                    astype(image.dtype))
        else:
            raise ValueError

    if args.visualize:
        while True:
            try:
                i = input("Patch number (0-{}): ".format(pred.shape[0] - 1))
                if i == "exit":
                    exit(0)
                else:
                    i = int(i)

                fig, axes = plt.subplots(1, 3)
                axes[0].imshow(image[i, :, :, 0], cmap="Greys_r")
                axes[0].set_title("Original")
                axes[0].set_aspect('equal', adjustable='box')
                axes[1].imshow(noisy[i, :, :, 0], cmap="Greys_r")
                axes[1].set_title("Noisy (sigma = {})".format(args.sigma))
                axes[1].set_aspect('equal', adjustable='box')
                axes[2].imshow(pred[i, :, :, 0], cmap="Greys_r")
                axes[2].set_title("Restored")
                axes[2].set_aspect('equal', adjustable='box')

                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])

                plt.show()
            except Exception:
                continue


if __name__ == "__main__":
    main()
