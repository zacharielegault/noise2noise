from typing import Union
from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D,
                          Concatenate, BatchNormalization, Activation,
                          Dropout)
from keras.models import Model


def unet(shape=(None, None, 1), output_channels=1):
    """Build a U-Net model using the architecture proposed by Lehtinen
    et al. (https://arxiv.org/pdf/1803.04189).

    Args:
        shape: tuple. Input shape.
        output_channels: integer. Number of output channels.
    """
    input_layer = Input(shape=shape, name="input_layer")
    x = input_layer
    skips = []

    enc_conv = [2, 1, 1, 1, 1, 1]
    for e, conv in enumerate(enc_conv):
        if e != len(enc_conv) - 1:  # No skip of pool5
            skips.append(x)

        x = conv_block(x, conv, 48, name="enc{}".format(e))

        if e != len(enc_conv) - 1:  # No pooling after enc_conv6
            x = MaxPooling2D((2, 2), name="pool{}".format(e+1))(x)

    dec_filters = [96, 96, 96, 96, [64, 32]]
    for d, filt in enumerate(dec_filters):
        x = UpSampling2D((2, 2), name="up{}".format(e-d-1))(x)
        x = Concatenate(name="concat{}".format(e-d-1))([x, skips.pop()])
        x = conv_block(x, 2, filt, name="dec{}".format(e-d-1))

    output_layer = Conv2D(
        filters=output_channels,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer='he_uniform',
        name="output_layer")(x)

    model = Model(input_layer, output_layer)

    return model


def conv_block(x,
               layers: int,
               filters: Union[int, list, tuple],
               name: str,
               act="relu",
               use_bn=False,
               use_res=False,
               dropout=0.):
    """Build a convolution block. All convolutions have a (3, 3)
    kernel size, have `"same"` padding and use the `"he_uniform"`
    initializer.

    Args:
        x: input tensor.
        layers: integer. Number of convolution layers in the block.
        filters: integer, or list/tuple of integers. Number of
            filters for each convolution layer. If a list or a
            tuple, its length must match the number of layers.
        name: string. Name of the convolution block.
        act: activation function.
        use_bn: boolean. Whether or not to use batch normalization.
            If True, batch norm is applied after activation and
            before dropout.
        use_res: boolean. Whether or not to use residual
            connections. Connections skip the convolution,
            activation, batch norm and dropout steps
        dropout: float between 0 and 1. Dropout rate.
    """
    if isinstance(filters, int):
        filters = [filters]*layers
    elif isinstance(filters, (list, tuple)):
        assert len(filters) == layers
        assert all(isinstance(f, int) for f in filters)
    else:
        raise TypeError

    skip = x if use_res else None
    for i in range(layers):
        x = Conv2D(
            filters=filters[i],
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_uniform",
            name=name + "_conv{}".format(i))(x)
        x = Activation(act, name=name + "_act{}".format(i))(x)
        if use_bn:
            x = BatchNormalization(name=name + "_bn{}".format(i))(x)
        if dropout:
            x = Dropout(dropout, name=name + "_drop{}".format(i))(x)

    x = Concatenate(name=name + "_res")([x, skip]) if use_res else x
    return x
