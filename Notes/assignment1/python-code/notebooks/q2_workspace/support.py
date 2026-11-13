"""Provided Q2 support code. No changes and no Q1 imports.

The only dependencies are NumPy, the cnn_layers module, and the
Python standard library. All training and differentiation run in NumPy.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
import cnn_layers


class Conv2D:
    """Fixed 3x3, stride-1, pad-1 wrapper; He-normal weights, zero bias."""
    def __init__(self, in_channels: int, out_channels: int,
                 rng: np.random.Generator):
        self.weight = (rng.standard_normal((out_channels, in_channels, 3, 3))
                       * np.sqrt(2.0 / (in_channels * 9))).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)
        self.dweight = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)

    def forward(self, x):
        out, self.cache = cnn_layers.conv2d_forward(x, self.weight, self.bias)
        return out

    def backward(self, dout):
        dx, self.dweight, self.dbias = cnn_layers.conv2d_backward(dout, self.cache)
        return dx


class MaxPool2D:
    def forward(self, x):
        out, self.cache = cnn_layers.maxpool2d_forward(x)
        return out

    def backward(self, dout):
        return cnn_layers.maxpool2d_backward(dout, self.cache)


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, dout):
        return dout * self.mask


class GlobalMaxPool2D:
    """Spatial maximum per channel; first row-major maximum wins ties."""
    def forward(self, x):
        self.shape = x.shape
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        self.index = np.argmax(flat, axis=2)
        return np.take_along_axis(flat, self.index[..., None], axis=2)[..., 0]

    def backward(self, dout):
        n, c, h, w = self.shape
        dx = np.zeros((n, c, h * w), dtype=dout.dtype)
        np.put_along_axis(dx, self.index[..., None], dout[..., None], axis=2)
        return dx.reshape(self.shape)


class Linear:
    """Provided affine classifier; Glorot-normal weights, zero bias."""
    def __init__(self, in_features: int, out_features: int,
                 rng: np.random.Generator):
        self.weight = (rng.standard_normal((in_features, out_features))
                       * np.sqrt(2.0 / (in_features + out_features))).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.dweight = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)

    def forward(self, x):
        self.x = x
        return x @ self.weight + self.bias

    def backward(self, dout):
        self.dweight = self.x.T @ dout
        self.dbias = dout.sum(axis=0)
        return dout @ self.weight.T


class Sequential:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return raw logits (N,2); cache values for the next backward call."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Return input gradient and overwrite the current parameter gradients."""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def named_parameters(self):
        """Yield (name, parameter array, gradient array), in stable order."""
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):
                yield f'layer{i}.weight', layer.weight, layer.dweight
                yield f'layer{i}.bias', layer.bias, layer.dbias

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return independent copies of the trainable parameter arrays."""
        return {name: p.copy() for name, p, _ in self.named_parameters()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Copy matching parameters in place; return None."""
        expected = {name for name, _, _ in self.named_parameters()}
        if set(state) != expected:
            raise ValueError('Checkpoint keys do not match the model')
        for name, p, _ in self.named_parameters():
            a = np.asarray(state[name])
            if a.shape != p.shape:
                raise ValueError(f'Wrong checkpoint shape for {name}')
            p[...] = a


def cross_entropy(logits: np.ndarray, y: np.ndarray
                  ) -> tuple[float, np.ndarray]:
    """Return (mean cross-entropy, dlogits), with dlogits shape (N,2)."""
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    log_probs = z - np.log(exp.sum(axis=1, keepdims=True))
    loss = -float(log_probs[np.arange(len(y)), y].mean())
    dlogits = probabilities.copy()
    dlogits[np.arange(len(y)), y] -= 1
    dlogits /= len(y)
    return loss, dlogits


class Adam:
    """Provided Adam, no weight decay. Reset for each fresh training run."""
    def __init__(self, lr=0.003, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, model: Sequential) -> None:
        self.t += 1
        for name, p, g in model.named_parameters():
            if name not in self.m:
                self.m[name] = np.zeros_like(p)
                self.v[name] = np.zeros_like(p)
            m, v = self.m[name], self.v[name]
            m[...] = self.beta1 * m + (1 - self.beta1) * g
            v[...] = self.beta2 * v + (1 - self.beta2) * (g * g)
            mhat = m / (1 - self.beta1 ** self.t)
            vhat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def evaluate(model: Sequential, split: dict, batch_size: int = 128) -> dict:
    """Return metrics and per-image predictions WITHOUT parameter updates.

    Returns keys:
      loss: sample-weighted mean cross-entropy (float)
      accuracy: fraction correct (float)
      positive_recall: TP/(TP+FN), or NaN when there are no positives
      confusion_matrix: int64 (2,2), rows=true, columns=predicted
      logits: (N,2), in original split order
      predictions: int64 (N,), argmax(logits); ties choose class 0
    """
    x, y = split['x'], split['y']
    if len(y) == 0:
        raise ValueError('Cannot evaluate an empty split')
    outputs, loss_sum = [], 0.0
    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        logits = model.forward(x[start:end])
        loss, _ = cross_entropy(logits, y[start:end])
        loss_sum += loss * (end - start)
        outputs.append(logits.copy())
    logits = np.concatenate(outputs, axis=0)
    predictions = logits.argmax(axis=1).astype(np.int64)
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (y, predictions), 1)
    positives = int(cm[1].sum())
    return dict(loss=loss_sum / len(y), accuracy=float((predictions == y).mean()),
                positive_recall=float(cm[1, 1] / positives) if positives else float('nan'),
                confusion_matrix=cm, logits=logits, predictions=predictions)


def train(model: Sequential, train_split: dict, validation_split: dict,
          epochs: int = 30, batch_size: int = 64, lr: float = 0.003,
          shuffle_seed: int = 18, verbose: bool = True) -> tuple[list[dict], dict]:
    """Train ALL epochs and restore the best validation-loss checkpoint.

    Returns
    -------
    history : list of dicts, one per epoch, with epoch, train_loss,
        validation_loss, train_accuracy, validation_accuracy, epoch_seconds.
        Metrics are evaluated after each epoch, not averaged over updates.
    summary : dict with selected_epoch, selected_validation_loss,
        training_seconds. Earlier epochs win exact validation-loss ties.

    Does not read either test set. The model is changed in place and ends
    with its selected parameters. Loss gradients are averaged exactly once.
    """
    rng = np.random.default_rng(shuffle_seed)
    optimizer = Adam(lr=lr)
    history, best_loss, best_state, best_epoch = [], float('inf'), None, None
    started = time.perf_counter()
    x, y = train_split['x'], train_split['y']
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            idx = order[start:start + batch_size]
            logits = model.forward(x[idx])
            _, dlogits = cross_entropy(logits, y[idx])
            model.backward(dlogits)
            optimizer.step(model)
        tr = evaluate(model, train_split)
        va = evaluate(model, validation_split)
        record = dict(epoch=epoch, train_loss=tr['loss'], validation_loss=va['loss'],
                      train_accuracy=tr['accuracy'], validation_accuracy=va['accuracy'],
                      epoch_seconds=time.perf_counter() - epoch_started)
        history.append(record)
        if va['loss'] < best_loss:
            best_loss, best_epoch = va['loss'], epoch
            best_state = model.state_dict()
        if verbose:
            print(f"Epoch {epoch:02d}: train loss={tr['loss']:.4f}, "
                  f"val loss={va['loss']:.4f}, val acc={va['accuracy']:.3f}")
    if best_state is None:
        raise RuntimeError('No finite checkpoint was produced')
    model.load_state_dict(best_state)
    summary = dict(selected_epoch=best_epoch, selected_validation_loss=best_loss,
                   training_seconds=time.perf_counter() - started)
    return history, summary


def save_run(directory: str, model: Sequential, history: list[dict],
             summary: dict, evaluations: dict[str, dict], data: dict,
             config: dict) -> None:
    """Write checkpoint.npz, history.csv, metrics.json, predictions.npz, config.json.

    Return None. NPZ files contain arrays only, never pickled Python objects.
    """
    import csv
    import platform
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / 'checkpoint.npz', **model.state_dict())
    with (path / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metrics, arrays = dict(summary), {}
    for name, result in evaluations.items():
        metrics[name] = {k: result[k] for k in ['loss', 'accuracy', 'positive_recall']}
        metrics[name]['confusion_matrix'] = result['confusion_matrix'].tolist()
        for k in ['logits', 'predictions']:
            arrays[f'{name}_{k}'] = result[k]
        arrays[f'{name}_labels'] = data[name]['y']
        arrays[f'{name}_layout_id'] = data[name]['layout_id']
    np.savez_compressed(path / 'predictions.npz', **arrays)
    (path / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    config = dict(config, numpy_version=np.__version__, python_version=platform.python_version())
    (path / 'config.json').write_text(json.dumps(config, indent=2))
