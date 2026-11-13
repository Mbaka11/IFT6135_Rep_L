"""Q2 implementations: NumPy only, no automatic differentiation.

All spatial tensors use NCHW. Implement only the four functions below.
The geometry is fixed: 3x3 cross-correlation, stride 1, zero padding 1;
2x2 max pooling, stride 2, no padding. Do not generalize the API.

Inputs are finite float32 or float64 arrays. Inputs belonging to a call have
matching dtypes. Preserve that dtype; never modify input arrays in place.
Support arbitrary positive batch/channel sizes and H,W >= 2. Pooling requires
even H,W. Noncontiguous NumPy views are valid inputs. A cache is an opaque,
per-forward object: choose its structure, but do not use global state.
A backward call returns fresh gradients, does not update parameters, and does
not divide by the batch size. The incoming gradient already incorporates any
loss reduction. Separate outstanding caches must work independently.
"""
from typing import Any
import numpy as np


def conv2d_forward(x: np.ndarray, weight: np.ndarray,
                   bias: np.ndarray) -> tuple[np.ndarray, Any]:
    """Apply 3x3 cross-correlation with stride 1 and zero padding 1.

    Parameters
    ----------
    x : ndarray, shape (N, C_in, H, W)
    weight : ndarray, shape (C_out, C_in, 3, 3)
        Do not flip the kernel. All channels contribute to each output filter.
    bias : ndarray, shape (C_out,)
        Add one scalar to every output position in the corresponding channel.

    Returns
    -------
    out : ndarray, shape (N, C_out, H, W)
        Same floating dtype as x.
    cache : object
        Information your conv2d_backward needs. Its structure is your choice.
        Do not update weights or biases in this function.
    """
    N, C_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    # Padding of 1 on two sides of H and W
    X_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((N, C_out, H_in, W_in), dtype=x.dtype)
    
    # Loop over all the 9 positions of the kernel
    # Images, channels, height, width are together
    for i in range(3):
        for j in range(3):
            patch = X_pad[:, :, i:i+H_in, j:j+W_in]
            
            # n = image, c_out = output channel, c_in = input channel
            # Addition on the input channel dimension, broadcasting over the output channels
            out += np.einsum(
                'nchw,oc->nohw', 
                patch, 
                weight[:, :, i, j],
                optimize=True
            )
            
    # Add a single bias per output channel, broadcasting over the batch and spatial dimensions
    out += bias[None, :, None, None]
    
    cache = (X_pad, weight, x.shape)
    return out, cache
            
def conv2d_backward(dout: np.ndarray, cache: Any
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through the matching conv2d_forward call.

    Parameters
    ----------
    dout : ndarray, shape (N, C_out, H, W)
        Upstream derivative of a scalar loss with respect to the output.
    cache : object returned by your conv2d_forward

    Returns
    -------
    dx : ndarray, shape (N, C_in, H, W)
    dweight : ndarray, shape (C_out, C_in, 3, 3)
    dbias : ndarray, shape (C_out,)
        Sum contributions over the batch and spatial axes; do not average.
        Return (dx, dweight, dbias), in that order, in the input dtype.
        Overlapping receptive fields must accumulate their input gradients.
    """
    X_pad, weight, x_shape = cache
    N, C_in, H_in, W_in = x_shape
    
    dx_pad = np.zeros_like(X_pad)
    dweight = np.zeros_like(weight)
    # Biais on the output channel dimension, summing over the batch and spatial dimensions
    dbias = np.sum(dout, axis=(0, 2, 3))
    
    for i in range(3):
        for j in range(3):
            patch = X_pad[:, :, i:i+H_in, j:j+W_in]
            
            # Compute the gradient with respect to the weights
            dweight[:, :, i, j] = np.einsum(
                'nohw,nchw->oc',
                dout,
                patch,
                optimize=True
            )
            # Input gradient is computed by convolving the upstream gradient with the flipped kernel
            dx_pad[:, :, i:i+H_in, j:j+W_in] += np.einsum(
                'nohw,oc->nchw',
                dout,
                weight[:, :, i, j],
                optimize=True
            )
            
    # Remove the padding from the input gradient
    dx = dx_pad[:, :, 1:-1, 1:-1]
    return dx, dweight, dbias


def maxpool2d_forward(x: np.ndarray) -> tuple[np.ndarray, Any]:
    """Apply 2x2 max pooling with stride 2 and no padding.

    Parameters
    ----------
    x : ndarray, shape (N, C, H, W)
        H and W must be even. Values may be negative or repeated.

    Returns
    -------
    out : ndarray, shape (N, C, H//2, W//2), same dtype as x
    cache : object
        Information your maxpool2d_backward needs. At tied maxima, select
        only the first maximum in row-major order within the 2x2 window:
        top-left, top-right, bottom-left, bottom-right. Cache this selection.
    """
    # we need to reshape the 2x2 windows into this order : 
    # 0 : top-left, 1 : top-right, 2 : bottom-left, 3 : bottom-right
    # Slice go forward by 2, so stride is 2
    windows = np.stack([
        x[:, :, 0::2, 0::2],  # top-left
        x[:, :, 0::2, 1::2],  # top-right
        x[:, :, 1::2, 0::2],  # bottom-left
        x[:, :, 1::2, 1::2]   # bottom-right
    ], axis=-1)  # shape (N, C, H//2, W//2, 4)
    
    # Argmax to find the index of the maximum value in each 2x2 window
    max_indices = np.argmax(windows, axis=-1)  # shape (N, C, H//2, W//2)
    # Retrieve the maximum values using advanced indexing
    out = np.take_along_axis(
        windows, max_indices[..., np.newaxis], axis=-1).squeeze(-1)  # shape (N, C, H//2, W//2)
    
    # Keep the max_indices in the cache for backward pass
    cache = (x.shape, max_indices, x.dtype)
    return out, cache

def maxpool2d_backward(dout: np.ndarray, cache: Any) -> np.ndarray:
    """Route each upstream gradient to its selected maximum.

    Parameters
    ----------
    dout : ndarray, shape (N, C, H//2, W//2)
    cache : object returned by your maxpool2d_forward

    Returns
    -------
    dx : ndarray, shape (N, C, H, W), same dtype as the forward input
        One selected position receives the upstream gradient per window;
        all other positions receive zero. Do not split a tied gradient or
        send it to every tied maximum. Do not average or update any arrays.
    """
    input_shape, max_indices, dtype = cache
    dx = np.zeros(input_shape, dtype=dtype)
    
    for k in range(4):
        i, j = divmod(k, 2)  # Get the row and column offsets for the 2x2 window
        # One receive the upstream gradient if it was the max, otherwise zero
        dx[:, :, i::2, j::2] += (max_indices == k) * dout
    
    return dx
