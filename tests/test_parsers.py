import io

import numpy as np
import pandas as pd
from PIL import Image

from esp_data.io.parsers import (
    read_bytes_to_array,
    read_csv_from_bytes,
    read_image_from_bytes,
    read_mat_from_bytes,
    read_npy_from_bytes,
    read_npz_from_bytes,
)


def test_read_image_from_bytes():
    img = Image.new("RGB", (2, 2), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    arr = read_image_from_bytes(image_bytes)
    assert arr.shape == (2, 2, 3)


def test_read_npy_from_bytes():
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, data)
    npy_bytes = buf.getvalue()
    arr = read_npy_from_bytes(npy_bytes)
    assert np.array_equal(arr, data)


def test_read_npz_from_bytes():
    data_a = np.array([1, 2, 3])
    data_b = np.array([[4, 5], [6, 7]])
    buf = io.BytesIO()
    np.savez(buf, a=data_a, b=data_b)
    npz_bytes = buf.getvalue()
    arrays = read_npz_from_bytes(npz_bytes)
    assert np.array_equal(arrays["a"], data_a)
    assert np.array_equal(arrays["b"], data_b)


def test_read_mat_from_bytes(tmp_path):
    # Placeholder. In practice, prepare real .mat bytes or mock scipy.io.loadmat.
    from scipy.io import savemat

    tmp_file = tmp_path / "some_mat.mat"
    m1 = np.random.rand(5, 5)
    savemat(str(tmp_file), mdict={"data": m1})
    with open(tmp_file, "rb") as f:
        m2 = f.read()
    m2 = read_mat_from_bytes(m2)["data"]
    assert np.array_equal(m1, m2)


def test_read_csv_from_bytes():
    csv_data = "col1,col2\n1,2\n3,4\n"
    df = read_csv_from_bytes(csv_data.encode("utf-8"))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_read_bytes_to_array_image():
    img = Image.new("RGB", (2, 2), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    parsed = read_bytes_to_array(buf.getvalue(), extension="jpg")
    assert parsed.shape == (2, 2, 3)
