import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    channel_tile = nl.tile_size.pmax
    n_tiles_c_in = in_channels // channel_tile
    n_tiles_c_out = out_channels // channel_tile


    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for oc in nl.affine_range(n_tiles_c_out):

            oc_start = oc * channel_tile

            bias_tile = nl.ndarray(
                shape=(channel_tile, 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(dst=bias_tile, src=bias[oc_start : oc_start + channel_tile])

            for oh in nl.affine_range(out_height):
                conv_psum = nl.zeros((channel_tile, out_width), dtype=nl.float32, buffer=nl.psum)

                for fh in nl.affine_range(filter_height):
                    for fw in nl.affine_range(filter_width):
                        for ic in nl.affine_range(in_channels // channel_tile):
                            ic_start = ic * channel_tile

                            X_tile = nl.ndarray(
                                shape=(channel_tile, out_width),
                                dtype=X.dtype,
                                buffer=nl.sbuf
                            )
                            W_tile = nl.ndarray(
                                shape=(channel_tile, channel_tile),
                                dtype=W.dtype,
                                buffer=nl.sbuf
                            )
                            stationary_tile = nl.ndarray(
                                shape=(channel_tile, channel_tile),
                                dtype=W.dtype,
                                buffer=nl.sbuf,
                            )

                            nisa.dma_copy(dst=X_tile,
                                          src=X[b,
                                                ic_start : ic_start + channel_tile,
                                                oh + fh,
                                                fw : fw + out_width]
                            )
                            nisa.dma_copy(dst=W_tile,
                                          src=W[oc_start : oc_start + channel_tile,
                                                ic_start : ic_start + channel_tile,
                                                fh,
                                                fw]
                            )

                            stationary_tile[...] = nisa.tensor_copy(src=nisa.nc_transpose(W_tile))

                            conv_psum += nisa.nc_matmul(stationary_tile[...], X_tile[...])

                conv_row = nl.copy(conv_psum, dtype=nl.float32)
                conv_row = nisa.tensor_scalar(conv_row, nl.add, bias_tile)
                conv_row = nl.copy(conv_row, dtype=X.dtype)

                nisa.dma_copy(dst=X_out[b,
                                        oc_start : oc_start + channel_tile,
                                        oh,
                                        0 : out_pool_width],
                              src=conv_row)


    return X_out

