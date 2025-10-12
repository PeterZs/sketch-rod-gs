from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "making_correspond",
        [
            "cpp_src/making_correspond.cpp", 
            "cpp_src/exp.cpp"
        ],
        include_dirs=[
            pybind11.get_include()
        ],
        language="c++",
        cxx_std=17,
        extra_compile_args=['-O3', '-std=c++17', '-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name="making_correspond",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
