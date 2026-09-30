"""Guard against the EMNIST archive's label-sorted writer prefixes."""

import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

import msgpack
import numpy as np

from run_pilot import ClientData


def encoded(array):
    return msgpack.ExtType(1, msgpack.packb(
        (array.shape, array.dtype.str, array.tobytes()), use_bin_type=True))


class ClientSelectionTests(unittest.TestCase):
    def test_uniform_sampling_is_deterministic_and_avoids_prefix_skew(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "writers.sqlite"
            images = np.arange(100, dtype=np.float32)[:, None, None]
            images = np.broadcast_to(images / 100, (100, 28, 28)).copy()
            labels = np.repeat(np.arange(10, dtype=np.int64), 10)
            payload = msgpack.packb({"pixels": encoded(images),
                                     "label": encoded(labels)}, use_bin_type=True)
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE federated_data ("
                                   "client_id BLOB PRIMARY KEY, data BLOB, "
                                   "num_examples INTEGER)")
                connection.execute("INSERT INTO federated_data VALUES (?,?,?)",
                                   (b"writer", zlib.compress(payload), 100))
            prefix = ClientData(path, 16, selection="first")
            _, old_labels = prefix.get(b"writer")
            self.assertLessEqual(len(np.unique(old_labels)), 2)
            first = ClientData(path, 16, selection="uniform", sample_seed=7)
            second = ClientData(path, 16, selection="uniform", sample_seed=7)
            x, selected = first.get(b"writer")
            x2, selected2 = second.get(b"writer")
            np.testing.assert_array_equal(x, x2)
            np.testing.assert_array_equal(selected, selected2)
            self.assertEqual(len(np.unique(x[:, 0])), 16)
            self.assertGreaterEqual(len(np.unique(selected)), 5)
            ink = ClientData(path, 16, selection="uniform", sample_seed=7,
                             pixel_transform="ink")
            ink_x, ink_labels = ink.get(b"writer")
            np.testing.assert_array_equal(selected, ink_labels)
            np.testing.assert_allclose(ink_x[:, :-1], 1 - x[:, :-1])
            np.testing.assert_array_equal(ink_x[:, -1], x[:, -1])


if __name__ == "__main__":
    unittest.main()
