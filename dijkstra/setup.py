from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "dijkstra_cpp", 
        ["cpp_src/dijkstra.cpp", "cpp_src/exp.cpp"], 
        define_macros=[("VERSION_INFO", __version__)], 
        include_dirs=["eigen-3.4.0"], 
    )
]

setup(
    name="dijkstra_cpp", 
    version=__version__, 
    author="Haato Watanabe", 
    description="dijkstra implementation with cpp", 
    long_description="", 
    ext_modules=ext_modules, 
    extras_require={"test": "pytest"}, 
    cmdclass={"build_ext": build_ext}, 
    zip_safe=False, 
    python_requires=">=3.7", 
)
