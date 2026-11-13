"""Q2 model assembly."""
import numpy as np
from support import Conv2D, ReLU, MaxPool2D, GlobalMaxPool2D, Linear, Sequential


def build_model(seed: int = 17) -> Sequential:
    """Construct and return a NEW model with the specified layer order.

    Conv2D(1, 8) -> ReLU -> MaxPool2D -> Conv2D(8, 16) -> ReLU
    -> GlobalMaxPool2D -> Linear(16, 2).

    Pass the SAME local rng = np.random.default_rng(seed) to the three
    trainable layer constructors, in their forward order. All convolutions
    use 3x3 kernels, stride 1 and zero padding 1. MaxPool2D uses a 2x2
    window and stride 2. Other constructor arguments must remain defaults.

    Provided constructors:
      Conv2D(in_channels, out_channels, rng)
      ReLU(), MaxPool2D(), GlobalMaxPool2D()
      Linear(in_features, out_features, rng)
      Sequential(list_of_layers)

    Returns
    -------
    model : support.Sequential
        model.forward(x): (N, 1, 24, 24) -> (N, 2) raw logits.
        No final softmax. The provided loss applies it internally.
        The convolution and local pooling wrappers call cnn_layers.py.
    """
    rng = np.random.default_rng(seed)
    model = Sequential([
        Conv2D(1, 8, rng),              # (N, 1, 24, 24) -> (N, 8, 24, 24)
        ReLU(),
        MaxPool2D(),                    # (N, 8, 24, 24) -> (N, 8, 12, 12)
        Conv2D(8, 16, rng),             # (N, 8, 12, 12) -> (N, 16, 12, 12)
        ReLU(),
        GlobalMaxPool2D(),              # (N, 16, 12, 12) -> (N, 16)
        Linear(16, 2, rng)              # (N, 16) -> (N, 2)
    ]) 
    return model
