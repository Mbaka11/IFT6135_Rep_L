import numpy as np
from abc import ABC, abstractmethod


class Module(ABC):
    """Base class for every layer/model in this mini neural-network library."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Computes the layer's output from its input(s)."""
        pass

    def backward(self, *args, **kwargs):
        """Computes the gradient of the loss w.r.t. the layer's input(s)."""
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Collects every learnable (parameter, gradient) pair from this module
        and all of its sub-modules (attributes that are themselves `Module`s).

        Returns:
            list of (param, grad) tuples, where `param` and `grad` are numpy
            arrays of identical shape. Leaf modules with parameters (e.g.
            `Linear`) override this method to append their own tuples.
        """
        params = []
        for value in vars(self).values():
            if isinstance(value, Module):
                params.extend(value.parameters())
        return params


class Linear(Module):
    """Fully-connected (dense) layer computing Z = X @ W.T + b."""

    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features: number of input features D_in.
            out_features: number of output features D_out.
        """
        super().__init__()
        # Glorot/Xavier uniform initialization: U(-limit, limit)
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, size=(out_features, in_features))
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: input, shape (batch_size, in_features).

        Returns:
            Z: output, shape (batch_size, out_features).
        """
        ##CODE
        # TODO: compute Z = X @ W.T + b. Cache whatever you need (e.g. X)
        # on self for use in backward().
        self.X = X
        Z = np.dot(X, self.W.T) + self.b
        return Z
        ##END

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        Args:
            dZ: gradient of the loss w.r.t. this layer's output Z,
                shape (batch_size, out_features).

        Returns:
            dX: gradient of the loss w.r.t. this layer's input X,
                shape (batch_size, in_features).
        """
        ##CODE
        # TODO: compute and store self.dW and self.db (gradients of the loss
        # w.r.t. W and b), then return dX, the gradient w.r.t. the input.
        # Fill the gradient buffers in-place with existing gradients (don't reassign self.dW or self.db).
        self.dW[...] = np.dot(dZ.T, self.X)
        self.db[...] = np.sum(dZ, axis=0)
        # Gradient w.r.t. input X
        dZ = np.dot(dZ, self.W)
        return dZ
        ##END

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return super().parameters() + [(self.W, self.dW), (self.b, self.db)]


class Sigmoid(Module):
    """Element-wise sigmoid activation: Y = 1 / (1 + exp(-Z))."""

    def __init__(self):
        super().__init__()
        self.Y = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Args:
            Z: pre-activation input, any shape (S,).

        Returns:
            Y: activated output, shape (S,), same shape as Z.
        """
        ##CODE
        # TODO: compute Y = sigmoid(Z). Cache what you need for backward().
        # Use a numerically stable implementation of the sigmoid function to avoid overflow issues.
        e = np.exp(-np.abs(Z))
        self.Y = np.where(
            Z >= 0, 
            1 / (1 + e), 
            e / (1 + e))
        return self.Y
        ##END

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Args:
            dY: gradient of the loss w.r.t. this layer's output Y,
                shape (S,), same shape as the forward input.

        Returns:
            dZ: gradient of the loss w.r.t. this layer's input Z, shape (S,).
        """
        ##CODE
        # TODO: apply the chain rule through the sigmoid derivative.
        # The derivative of the sigmoid function is sigmoid(Z) * (1 - sigmoid(Z)).
        dZ = dY * self.Y * (1 - self.Y)
        return dZ
        ##END


class ReLU(Module):
    """Element-wise ReLU activation: Y = max(Z, 0)."""

    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Args:
            Z: pre-activation input, any shape (S,).

        Returns:
            Y: activated output, shape (S,), same shape as Z.
        """
        ##CODE
        # TODO: compute Y = max(Z, 0). Cache what you need for backward().
        # The mask will be used to identify where Z is greater than 0 for the backward pass.
        self.mask = Z > 0
        Y = np.maximum(Z, 0)
        return Y
        ##END

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Args:
            dY: gradient of the loss w.r.t. this layer's output Y,
                shape (S,), same shape as the forward input.

        Returns:
            dZ: gradient of the loss w.r.t. this layer's input Z, shape (S,).
        """
        ##CODE
        # TODO: apply the chain rule through the ReLU derivative.
        # The derivative of ReLU is 1 for Z > 0 and 0 otherwise, which is captured by the mask.
        dZ = dY * self.mask
        return dZ
        ##END


class Tanh(Module):
    """Element-wise hyperbolic tangent activation: Y = tanh(Z)."""

    def __init__(self):
        super().__init__()
        self.Y = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Args:
            Z: pre-activation input, any shape (S,).

        Returns:
            Y: activated output, shape (S,), same shape as Z.
        """
        ##CODE
        # TODO: compute Y = tanh(Z). Cache what you need for backward().
        self.Y = np.tanh(Z)
        return self.Y
        ##END

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Args:
            dY: gradient of the loss w.r.t. this layer's output Y,
                shape (S,), same shape as the forward input.

        Returns:
            dZ: gradient of the loss w.r.t. this layer's input Z, shape (S,).
        """
        ##CODE
        # TODO: apply the chain rule through the tanh derivative.
        # The derivative of tanh is 1 - tanh(Z)^2, which can be computed using the cached output Y.
        dZ = dY * (1 - self.Y ** 2)
        return dZ
        ##END


class MSE(Module):
    """Mean squared error loss: L = mean((Y - Yt) ** 2)."""

    def __init__(self):
        super().__init__()
        self.diff = None
        self.n = None

    def forward(self, Y: np.ndarray, Yt: np.ndarray) -> float:
        """
        Args:
            Y: model predictions, shape (batch_size, out_features).
            Yt: targets, shape (batch_size, out_features), same shape as Y.

        Returns:
            loss: scalar mean squared error.
        """
        ##CODE
        # TODO: compute the scalar MSE loss. Cache what you need for backward().
        self.diff = Y - Yt
        # Total number of elements in the difference
        self.n = self.diff.size 
        # Compute the mean squared error loss
        loss = np.mean(self.diff ** 2)
        return loss
        ##END

    def backward(self) -> np.ndarray:
        """
        Returns:
            dY: gradient of the loss w.r.t. predictions Y,
                shape (batch_size, out_features), same shape as Y.
        """
        ##CODE
        # TODO: return the gradient of the MSE loss w.r.t. Y.
        # The gradient of the MSE loss w.r.t. Y is (2/n) * (Y - Yt), where n is the number of elements.
        dY = (2 / self.n) * self.diff
        return dY
        ##END


class MAE(Module):
    """Mean absolute error loss: L = mean(|Y - Yt|)."""

    def __init__(self):
        super().__init__()
        self.diff = None
        self.n = None

    def forward(self, Y: np.ndarray, Yt: np.ndarray) -> float:
        """
        Args:
            Y: model predictions, shape (batch_size, out_features).
            Yt: targets, shape (batch_size, out_features), same shape as Y.

        Returns:
            loss: scalar mean absolute error.
        """
        ##CODE
        # TODO: compute the scalar MAE loss. Cache what you need for backward().
        self.diff = Y - Yt
        # Total number of elements in the difference
        self.n = self.diff.size
        # Compute the mean absolute error loss
        loss = np.mean(np.abs(self.diff))
        return loss
        ##END

    def backward(self) -> np.ndarray:
        """
        Returns:
            dY: gradient of the loss w.r.t. predictions Y,
                shape (batch_size, out_features), same shape as Y.
        """
        ##CODE
        # TODO: return the gradient of the MAE loss w.r.t. Y (the MAE
        # derivative is undefined at 0; treat it as 0 there).
        dY = np.sign(self.diff) / self.n
        return dY
        ##END
