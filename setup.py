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
    # Env vars
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(env_var)
        if val and os.path.isdir(val):
            return val

    # nvcc on PATH
    nvcc = shutil.which("nvcc")
    if nvcc:
        # nvcc is typically at <cuda_home>/bin/nvcc
        cuda_home = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
        if os.path.isdir(os.path.join(cuda_home, "include")):
            return cuda_home

    # Common paths
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


def _cuda_is_available():
    """Check if CUDA toolkit is usable (nvcc compiles a trivial program)."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return False
    tmp_src = ""
    tmp_out = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
            f.write("int main() { return 0; }\n")
            f.flush()
            tmp_src = f.name
        tmp_out = tmp_src.replace(".cu", ".out")
        result = subprocess.run(
            [nvcc, tmp_src, "-o", tmp_out],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        for p in (tmp_src, tmp_out):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Custom build_ext that compiles .cu files with nvcc
# ---------------------------------------------------------------------------

class CUDABuildExt(build_ext):
    """Custom build_ext that:
    1. Copies .cpp CUDA source to .cu so nvcc treats it as CUDA
    2. Compiles .cu sources with nvcc instead of the system C compiler
    3. Falls back gracefully if compilation fails
    """

    def build_extensions(self):
        for ext in self.extensions:
            if not getattr(ext, "_is_cuda", False):
                continue

            # Rewrite sources: rename .cpp → .cu in a temp location
            new_sources = []
            for src in ext.sources:
                if src.endswith(".cpp") and "_cuda_dtw" in src:
                    cu_src = src.rsplit(".cpp", 1)[0] + ".cu"
                    # Copy to .cu so nvcc picks it up
                    cu_path = os.path.join(self.build_temp, os.path.basename(cu_src))
                    os.makedirs(os.path.dirname(cu_path) or ".", exist_ok=True)
                    shutil.copy2(src, cu_path)
                    new_sources.append(cu_path)
                else:
                    new_sources.append(src)
            ext.sources = new_sources

        try:
            super().build_extensions()
        except Exception as e:
            print(f"\n⚠️  CUDA extension build failed: {e}")
            print("   Falling back to CPU-only installation.\n")
            # Remove CUDA extensions and proceed
            self.extensions = [
                ext for ext in self.extensions if not getattr(ext, "_is_cuda", False)
            ]
            if self.extensions:
                super().build_extensions()

    def build_extension(self, ext):
        if not getattr(ext, "_is_cuda", False):
            super().build_extension(ext)
            return

        # Use nvcc as the compiler for CUDA extensions
        cuda_home = _find_cuda_home()
        nvcc = shutil.which("nvcc") or os.path.join(cuda_home, "bin", "nvcc")

        # Gather include dirs, library dirs
        include_dirs = list(ext.include_dirs or [])
        library_dirs = list(ext.library_dirs or [])
        if cuda_home:
            include_dirs.append(os.path.join(cuda_home, "include"))
            lib_candidates = [
                os.path.join(cuda_home, "lib64"),
                os.path.join(cuda_home, "lib"),
            ]
            for ldir in lib_candidates:
                if os.path.isdir(ldir):
                    library_dirs.append(ldir)
                    break

        # Build output path
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)

        # Compile each source into an object file
        objects = []
        for src in ext.sources:
            obj = src + ".o"
            compile_cmd = [
                nvcc,
                "-std=c++11",
                "-O3",
                "-Xcompiler", "-fPIC",
                "--shared",
                "-c", src,
                "-o", obj,
            ]
            # Add include directories
            for inc in include_dirs:
                compile_cmd.extend(["-I", inc])
            # Add Python include
            compile_cmd.extend(["-I", self._get_python_include()])
            # Add numpy include
            compile_cmd.extend(["-I", self._get_numpy_include()])
            # Add CUDA source directory for local headers
            cuda_src_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "baleen", "_cuda_dtw",
            )
            compile_cmd.extend(["-I", cuda_src_dir])

            print(f"  nvcc compile: {' '.join(compile_cmd)}")
            subprocess.check_call(compile_cmd)
            objects.append(obj)

        # Link into shared library
        link_cmd = [
            nvcc,
            "--shared",
            "-o", ext_path,
        ] + objects

        for ldir in library_dirs:
            link_cmd.extend(["-L", ldir])
        for lib in (ext.libraries or []):
            link_cmd.extend(["-l", lib])

        # Link against Python library
        link_cmd.extend(["-Xcompiler", f"-undefined,dynamic_lookup"])

        print(f"  nvcc link: {' '.join(link_cmd)}")
        subprocess.check_call(link_cmd)

    @staticmethod
    def _get_python_include():
        from sysconfig import get_paths
        return get_paths()["include"]

    @staticmethod
    def _get_numpy_include():
        import numpy
        return numpy.get_include()


# ---------------------------------------------------------------------------
# Extension definition
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
    # Tag so our custom build_ext knows to handle it specially
    ext._is_cuda = True
    return ext


# ---------------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------------

def _should_build_cuda():
    """Decide whether to attempt CUDA extension build."""
    # User explicitly disabled
    if os.environ.get("BALEEN_NO_CUDA", "").strip() in ("1", "true", "yes"):
        print("ℹ️  CUDA extension disabled by BALEEN_NO_CUDA env var.")
        return False

    # Check for CUDA toolkit
    if not _cuda_is_available():
        print("ℹ️  CUDA toolkit not found. Installing without GPU acceleration.")
        return False

    return True


ext_modules = []
cmdclass = {}

if _should_build_cuda():
    ext_modules.append(_make_cuda_extension())
    cmdclass["build_ext"] = CUDABuildExt
    print("🔧 CUDA extension will be built.")

setup(
    name="baleen",
    version="0.1.0",
    description="CUDA-accelerated Dynamic Time Warping (DTW) for time series analysis",
    author="Logan",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
