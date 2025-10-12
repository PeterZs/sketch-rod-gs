from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
        "moving_primitives",
        ["cpp_src/moving_primitives.cpp", "cpp_src/exp.cpp"],
    ),
]

setup(
    name="moving_primitives",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
