import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "preprocess_cpp",
        [
            "cpp_src/exp.cpp",
            "cpp_src/preprocess.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "/usr/include/eigen3",  # Path for Eigen
        ],
        language="c++",
        cxx_std=17,
        extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="preprocess_cpp",
    version=__version__,
    author="Haato Watanabe",
    description="A test project using pybind11",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
