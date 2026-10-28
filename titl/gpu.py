"""Preload the pip-installed cuBLAS and cuDNN so ctranslate2 can find them."""
from __future__ import annotations

import ctypes
import glob
import importlib.util
import os
import sys


def preload_cuda_libs() -> list[str]:
    if sys.platform != "linux":
        return []
    loaded: list[str] = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if not spec or not spec.submodule_search_locations:
            continue
        libdir = os.path.join(list(spec.submodule_search_locations)[0], "lib")
        for pat in ("libcublasLt.so*", "libcublas.so*", "libcudnn.so*"):
            for f in sorted(glob.glob(os.path.join(libdir, pat))):
                if os.path.islink(f) and not f.endswith(".so"):
                    continue
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(f))
                    break
                except OSError:
                    continue
    return loaded
