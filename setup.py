"""
Baleen - CUDA-accelerated Dynamic Time Warping

Build modes:
  - With CUDA (GPU): Automatically detected if nvcc is on PATH.
                     Set CUDA_HOME or CUDA_PATH to override CUDA toolkit location.
                     Force skip with: BALEEN_NO_CUDA=1 pip install .
  - Without CUDA:   Pure-Python install. The _cuda_dtw module will still import
                     but dtw_distance()/dtw_pairwise() raise RuntimeError.
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


# ---------------------------------------------------------------------------
# CUDA detection helpers
# ---------------------------------------------------------------------------

def _find_cuda_home():
    """Find CUDA toolkit installation directory.

    Search order:
      1. CUDA_HOME environment variable
      2. CUDA_PATH environment variable
      3. nvcc location on PATH (derive from its parent)
      4. Common installation paths
    """
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(env_var)
        if val and os.path.isdir(val):
            return val

    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_home = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
        if os.path.isdir(os.path.join(cuda_home, "include")):
            return cuda_home

    common_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-11",
        "/opt/cuda",
    ]
    for path in common_paths:
        if os.path.isdir(path):
            return path

    return None


def _nvcc_is_available():
    """Quick check: is nvcc on PATH or in CUDA_HOME?"""
    if shutil.which("nvcc"):
        return True
    cuda_home = _find_cuda_home()
    if cuda_home:
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        return os.path.isfile(nvcc_path)
    return False


def _get_nvcc():
    """Return the full path to nvcc, or None."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    cuda_home = _find_cuda_home()
    if cuda_home:
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.isfile(nvcc_path):
            return nvcc_path
    return None


# ---------------------------------------------------------------------------
# Custom build_ext that compiles CUDA extensions with nvcc
# ---------------------------------------------------------------------------

class CUDABuildExt(build_ext):
    """Custom build_ext that:
    1. Detects CUDA availability at build time (not setup time)
    2. Copies .cpp CUDA source to .cu so nvcc treats it as CUDA
    3. Compiles .cu sources with nvcc
    4. Falls back gracefully if CUDA is missing or compilation fails
    """

    def build_extensions(self):
        # ── Gate: should we even try? ──
        cuda_exts = [e for e in self.extensions if getattr(e, "_is_cuda", False)]
        other_exts = [e for e in self.extensions if not getattr(e, "_is_cuda", False)]

        if os.environ.get("BALEEN_NO_CUDA", "").strip() in ("1", "true", "yes"):
            print("\nℹ️  CUDA build disabled by BALEEN_NO_CUDA env var.")
            self.extensions = other_exts
            if self.extensions:
                super().build_extensions()
            return

        if not _nvcc_is_available():
            print("\nℹ️  nvcc not found. Skipping CUDA extension (CPU-only install).")
            self.extensions = other_exts
            if self.extensions:
                super().build_extensions()
            return

        # ── Prepare .cpp → .cu copies ──
        for ext in cuda_exts:
            new_sources = []
            for src in ext.sources:
                if src.endswith(".cpp") and "_cuda_dtw" in src:
                    os.makedirs(self.build_temp, exist_ok=True)
                    cu_path = os.path.join(self.build_temp, os.path.basename(
                        src.rsplit(".cpp", 1)[0] + ".cu"
                    ))
                    shutil.copy2(src, cu_path)
                    new_sources.append(cu_path)
                else:
                    new_sources.append(src)
            ext.sources = new_sources

        # ── Build, with graceful fallback ──
        try:
            super().build_extensions()
        except Exception as e:
            print(f"\n⚠️  CUDA extension build failed: {e}")
            print("   Falling back to CPU-only installation.\n")
            self.extensions = other_exts
            if self.extensions:
                super().build_extensions()

    def build_extension(self, ext):
        if not getattr(ext, "_is_cuda", False):
            super().build_extension(ext)
            return

        nvcc = _get_nvcc()
        if not nvcc:
            raise RuntimeError("nvcc not found")

        cuda_home = _find_cuda_home()

        # ── Include / library directories ──
        include_dirs = list(ext.include_dirs or [])
        library_dirs = list(ext.library_dirs or [])

        if cuda_home:
            include_dirs.append(os.path.join(cuda_home, "include"))
            for lib_subdir in ("lib64", "lib"):
                ldir = os.path.join(cuda_home, lib_subdir)
                if os.path.isdir(ldir):
                    library_dirs.append(ldir)
                    break

        # Local CUDA headers (dtw.hpp, cuda_utils.hpp, etc.)
        cuda_src_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "baleen", "_cuda_dtw",
        )
        include_dirs.append(cuda_src_dir)

        # Python + NumPy includes
        from sysconfig import get_paths as _get_paths
        include_dirs.append(_get_paths()["include"])
        try:
            import numpy
            include_dirs.append(numpy.get_include())
        except ImportError:
            raise RuntimeError("numpy is required to build the CUDA extension")

        # ── Output path ──
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)

        # ── Compile each source ──
        objects = []
        for src in ext.sources:
            obj = src + ".o"
            compile_cmd = [
                nvcc,
                "-std=c++11",
                "-O3",
                "-Xcompiler", "-fPIC",
                "-c", src,
                "-o", obj,
            ]
            for inc in include_dirs:
                compile_cmd.extend(["-I", inc])

            print(f"  [baleen] nvcc compile: {os.path.basename(src)}")
            subprocess.check_call(compile_cmd)
            objects.append(obj)

        # ── Link ──
        link_cmd = [nvcc, "--shared", "-o", ext_path] + objects

        for ldir in library_dirs:
            link_cmd.extend(["-L", ldir])
        for lib in (ext.libraries or []):
            link_cmd.extend(["-l", lib])

        # Platform-specific linker flags
        if platform.system() == "Darwin":
            link_cmd.extend(["-Xcompiler", "-undefined,dynamic_lookup"])

        print(f"  [baleen] nvcc link: {os.path.basename(ext_path)}")
        subprocess.check_call(link_cmd)


# ---------------------------------------------------------------------------
# Extension definition — always declared, build_ext decides whether to build
# ---------------------------------------------------------------------------

def _make_cuda_extension():
    """Create the CUDA extension module definition."""
    cuda_src_dir = os.path.join("baleen", "_cuda_dtw")
    ext = Extension(
        name="baleen._cuda_dtw._cuda_dtw",
        sources=[
            os.path.join(cuda_src_dir, "dtw_api.cpp"),
            os.path.join(cuda_src_dir, "multithreading.cpp"),
        ],
        libraries=["cudart"],
        language="c++",
    )
    ext._is_cuda = True  # type: ignore[attr-defined]
    return ext


# ---------------------------------------------------------------------------
# Always include the CUDA extension; CUDABuildExt skips it if nvcc is absent
# ---------------------------------------------------------------------------

setup(
    name="baleen",
    version="0.1.0",
    description="CUDA-accelerated DTW and nanopore signal analysis pipeline",
    author="Logan",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tslearn",
        "pysam",
        "scipy",
        "tqdm",
        "pandas",
    ],
    entry_points={
        "console_scripts": ["baleen=baleen.cli:main"],
    },
    ext_modules=[_make_cuda_extension()],
    cmdclass={"build_ext": CUDABuildExt},
    zip_safe=False,
)
