from __future__ import annotations

import numpy as np
import pytest
import torch

from matgl import set_default_dtype


def test_set_default_dtype():
    set_default_dtype("float", 32)
    assert torch.get_default_dtype() == torch.float32
    assert np.dtype("float32") == np.float32


def test_set_default_dtype_invalid_size():
    with pytest.raises(ValueError, match="Invalid dtype size"):
        set_default_dtype("float", 128)
    set_default_dtype("float", 32)


@pytest.mark.skipif(torch.cuda.is_available(), reason="float16 is supported on CUDA")
def test_set_default_dtype_exception():
    with pytest.raises(Exception, match=r"torch.float16 is not supported for M3GNet"):
        set_default_dtype("float", 16)
    set_default_dtype("float", 32)
