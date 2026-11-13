"""Visible smoke tests and numerical gradient checks.

Run: python public_tests.py
"""
from __future__ import annotations
import unittest
import numpy as np
import cnn_layers as layers


def finite_difference(f, array: np.ndarray, index: tuple, eps: float = 1e-6) -> float:
    """Return the central-difference derivative of scalar f(array) at index.

    Uses two independent copies; does not modify array. Use float64.
    This is a debugging calculation, NOT the backward used in training.
    """
    plus, minus = array.copy(), array.copy()
    plus[index] += eps
    minus[index] -= eps
    return float((f(plus) - f(minus)) / (2 * eps))


def gradient_check() -> dict[str, dict[str, float]]:
    """Return max absolute/scaled errors for dx, dweight, dbias, pool_dx.

    Checks seeded float64 inputs with a scalar objective sum(out * upstream).
    Uses selected entries, eps=1e-6, atol=2e-6 and rtol=2e-4.
    Raises AssertionError if a checked entry fails. Pooling inputs have
    unique maxima; tied points are tested separately, not differentiated.
    """
    rng = np.random.default_rng(107)
    x = rng.normal(size=(2, 2, 4, 6))
    w = rng.normal(size=(3, 2, 3, 3))
    b = rng.normal(size=3)
    out, cache = layers.conv2d_forward(x, w, b)
    upstream = rng.normal(size=out.shape)
    dx, dw, db = layers.conv2d_backward(upstream, cache)
    records = {}
    for name, a, analytical, f in [
        ('dx', x, dx, lambda z: np.sum(layers.conv2d_forward(z, w, b)[0] * upstream)),
        ('dweight', w, dw, lambda z: np.sum(layers.conv2d_forward(x, z, b)[0] * upstream)),
        ('dbias', b, db, lambda z: np.sum(layers.conv2d_forward(x, w, z)[0] * upstream)),
    ]:
        indices = rng.choice(a.size, size=min(12, a.size), replace=False)
        numerical = np.array([finite_difference(f, a, np.unravel_index(i, a.shape)) for i in indices])
        exact = analytical.ravel()[indices]
        np.testing.assert_allclose(exact, numerical, atol=2e-6, rtol=2e-4)
        errors = np.abs(exact - numerical)
        records[name] = dict(max_absolute_error=float(errors.max()),
                             max_scaled_error=float((errors / np.maximum(1, np.abs(exact) + np.abs(numerical))).max()))
    # Distinct ramp values in each window keep the maximizing index stable.
    xpool = rng.permutation(np.arange(48, dtype=np.float64)).reshape(1, 2, 4, 6) / 7
    pooled, cache = layers.maxpool2d_forward(xpool)
    dp = rng.normal(size=pooled.shape)
    analytical = layers.maxpool2d_backward(dp, cache)
    f = lambda z: np.sum(layers.maxpool2d_forward(z)[0] * dp)
    numerical = np.array([finite_difference(f, xpool, idx) for idx in np.ndindex(xpool.shape)]).reshape(xpool.shape)
    np.testing.assert_allclose(analytical, numerical, atol=2e-6, rtol=2e-4)
    errors = np.abs(analytical - numerical)
    records['pool_dx'] = dict(max_absolute_error=float(errors.max()),
                              max_scaled_error=float((errors / np.maximum(1, np.abs(analytical) + np.abs(numerical))).max()))
    return records


class PublicTests(unittest.TestCase):
    def test_conv_ones_and_padding(self):
        x = np.arange(1, 10, dtype=np.float64).reshape(1, 1, 3, 3)
        out, _ = layers.conv2d_forward(x, np.ones((1, 1, 3, 3)), np.array([0.5]))
        expected = np.array([[12, 21, 16], [27, 45, 33], [24, 39, 28]], dtype=np.float64) + .5
        np.testing.assert_allclose(out, expected[None, None])

    def test_cross_correlation_not_flipped(self):
        x = np.zeros((1, 1, 4, 4), dtype=np.float64)
        x[0, 0, 1, 1] = 1
        w = np.arange(9, dtype=np.float64).reshape(1, 1, 3, 3)
        out, _ = layers.conv2d_forward(x, w, np.zeros(1))
        expected = np.array([[8, 7, 6, 0], [5, 4, 3, 0], [2, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float64)
        np.testing.assert_allclose(out, expected[None, None])

    def test_pool_values_and_backward(self):
        x = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)
        out, cache = layers.maxpool2d_forward(x)
        np.testing.assert_array_equal(out, np.array([[[[5, 7], [13, 15]]]]))
        dx = layers.maxpool2d_backward(np.ones_like(out), cache)
        expected = np.zeros_like(x)
        expected[:, :, 1::2, 1::2] = 1
        np.testing.assert_array_equal(dx, expected)

    def test_pool_ties(self):
        x = np.full((1, 1, 2, 2), -3.0)
        out, cache = layers.maxpool2d_forward(x)
        dx = layers.maxpool2d_backward(np.array([[[[2.0]]]]), cache)
        np.testing.assert_array_equal(out, np.array([[[[-3.0]]]]))
        np.testing.assert_array_equal(dx, np.array([[[[2., 0.], [0., 0.]]]]))

    def test_float32_and_rectangular_shapes(self):
        x = np.ones((2, 2, 4, 6), dtype=np.float32)
        w = np.ones((3, 2, 3, 3), dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        out, cache = layers.conv2d_forward(x, w, b)
        dx, dw, db = layers.conv2d_backward(np.ones_like(out), cache)
        for a, shape in [(out, (2,3,4,6)), (dx, x.shape), (dw, w.shape), (db, b.shape)]:
            self.assertEqual(a.shape, shape)
            self.assertEqual(a.dtype, np.float32)

    def test_numerical_gradients(self):
        gradient_check()


def run_public_tests() -> unittest.result.TestResult:
    """Run all layer tests; return a unittest TestResult."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PublicTests)
    return unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    result = run_public_tests()
    raise SystemExit(0 if result.wasSuccessful() else 1)
