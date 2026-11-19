"""metalcuda — run CUDA code on Apple GPUs from Python.

Two integration modes:

1. **In-process** (no relaunch)::

       import metalcuda
       metalcuda.enable()            # patches ctypes / ctypes.util.find_library
       import ctypes
       cuda = ctypes.CDLL("libcuda.so.1")   # -> libmetalcuda.dylib

   Every later ``ctypes.CDLL`` / ``find_library`` request for an NVIDIA
   driver or runtime library (libcuda.so.1, libcudart.so.12, nvcuda.dll, ...)
   resolves to libmetalcuda.dylib, which is also loaded RTLD_GLOBAL so code
   that looks symbols up with ``dlsym(RTLD_DEFAULT, ...)`` finds them.

2. **Injected** (unmodified programs)::

       python -m metalcuda your_script.py [args...]

   re-executes the interpreter with ``DYLD_INSERT_LIBRARIES`` pointing at
   libmetalcuda.dylib, whose dyld interposer redirects *every* ``dlopen`` of a
   CUDA library in the process — including ones made from C extensions — and
   with ``DYLD_LIBRARY_PATH`` exposing libcuda.dylib / libcudart.dylib for
   extensions linked against them.

Note: binaries protected by System Integrity Protection (e.g. /usr/bin/python3)
ignore DYLD_* variables; use a Homebrew / pyenv / venv interpreter.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import sys

__all__ = ["library_path", "library_dir", "enable", "is_enabled", "injected", "environment", "run"]
__version__ = "1.0.0"

_CUDA_LIB_RE = re.compile(
    r"(^|/)(libcuda(\.so(\.\d+)*|\.dylib|\.\d+\.dylib)|libcudart(\.so(\.\d+)*|\.dylib|\.\d+\.dylib)|"
    r"nvcuda(\.dll)?|cudart(64|32)_\d+\.dll|CUDA\.framework/.*CUDA)$"
)
_FIND_NAMES = {"cuda", "cudart", "nvcuda", "libcuda", "libcudart", "cudart64_12", "cudart64_110"}

_lib = None
_orig_cdll_init = None
_orig_find_library = None


def library_dir() -> str:
    """Directory holding libmetalcuda.dylib and the libcuda/libcudart names."""
    cands = []
    if os.environ.get("METALCUDA_LIB"):
        cands.append(os.path.dirname(os.path.abspath(os.environ["METALCUDA_LIB"])))
    here = os.path.dirname(os.path.abspath(__file__))
    cands += [os.path.join(here, "_lib"), os.path.normpath(os.path.join(here, "..", "..", "build", "lib"))]
    for d in cands:
        if os.path.exists(os.path.join(d, "libmetalcuda.dylib")):
            return d
    raise FileNotFoundError("libmetalcuda.dylib not found (build it with `make`, or set METALCUDA_LIB)")


def library_path() -> str:
    if os.environ.get("METALCUDA_LIB"):
        return os.path.abspath(os.environ["METALCUDA_LIB"])
    return os.path.join(library_dir(), "libmetalcuda.dylib")


def is_cuda_library_name(name) -> bool:
    if not isinstance(name, (str, bytes, os.PathLike)):
        return False
    name = os.fsdecode(name)
    return bool(_CUDA_LIB_RE.search(name))


def injected() -> bool:
    """True if libmetalcuda was injected with DYLD_INSERT_LIBRARIES at startup."""
    libsys = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    libsys._dyld_image_count.restype = ctypes.c_uint32
    libsys._dyld_get_image_name.restype = ctypes.c_char_p
    libsys._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    for i in range(libsys._dyld_image_count()):
        n = libsys._dyld_get_image_name(i)
        if n and n.endswith(b"libmetalcuda.dylib"):
            return "metalcuda" in os.environ.get("DYLD_INSERT_LIBRARIES", "")
    return False


def is_enabled() -> bool:
    return _orig_cdll_init is not None or injected()


def enable(verbose: bool = False) -> ctypes.CDLL:
    """Load libmetalcuda globally and redirect ctypes lookups of CUDA libraries to it."""
    global _lib, _orig_cdll_init, _orig_find_library
    path = library_path()
    if _lib is None:
        _lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    if _orig_cdll_init is None:
        _orig_cdll_init = ctypes.CDLL.__init__

        def _cdll_init(self, name, *args, **kwargs):
            if name is not None and is_cuda_library_name(name):
                if verbose:
                    print(f"[metalcuda] ctypes.CDLL({name!r}) -> {path}", file=sys.stderr)
                name = path
            return _orig_cdll_init(self, name, *args, **kwargs)

        ctypes.CDLL.__init__ = _cdll_init

        _orig_find_library = ctypes.util.find_library

        def _find_library(name):
            if name in _FIND_NAMES or is_cuda_library_name(name):
                return path
            return _orig_find_library(name)

        ctypes.util.find_library = _find_library
    # child processes inherit the redirection
    os.environ.update(environment(os.environ))
    return _lib


def disable() -> None:
    global _orig_cdll_init, _orig_find_library
    if _orig_cdll_init is not None:
        ctypes.CDLL.__init__ = _orig_cdll_init
        ctypes.util.find_library = _orig_find_library
        _orig_cdll_init = _orig_find_library = None


def environment(base=None) -> dict:
    """Environment variables that inject MetaCuda into a new process."""
    env = dict(base if base is not None else os.environ)
    lib, d = library_path(), library_dir()
    ins = [p for p in env.get("DYLD_INSERT_LIBRARIES", "").split(":") if p and "libmetalcuda" not in p]
    env["DYLD_INSERT_LIBRARIES"] = ":".join([lib] + ins)
    paths = [p for p in env.get("DYLD_LIBRARY_PATH", "").split(":") if p and p != d]
    env["DYLD_LIBRARY_PATH"] = ":".join([d] + paths)
    return env


def _sip_protected(exe: str) -> bool:
    real = os.path.realpath(exe)
    return real.startswith(("/usr/bin/", "/bin/", "/sbin/", "/System/", "/usr/libexec/"))


def run(argv, env=None):
    """Replace the current process with `argv` running under MetaCuda injection."""
    exe = argv[0]
    if _sip_protected(exe if os.path.isabs(exe) else (shutil_which(exe) or exe)):
        print(f"[metalcuda] warning: {exe} is SIP-protected; DYLD_INSERT_LIBRARIES will be ignored. "
              "Use a Homebrew/pyenv/venv Python.", file=sys.stderr)
    os.execvpe(exe, argv, environment(env))


def shutil_which(name):
    import shutil
    return shutil.which(name)
