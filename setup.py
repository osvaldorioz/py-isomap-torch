from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import pybind11

#pip3.12 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
#pip3.12 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

#python3.12 setup.py build_ext --inplace

setup(
    name="isomap_module",
    ext_modules=[
        CppExtension(
            "isomap_module",
            ["isomap.cpp"],
            include_dirs=[pybind11.get_include()],
            extra_compile_args=["-std=c++17", "-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
