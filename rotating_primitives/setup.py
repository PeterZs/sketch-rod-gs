from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        "rotating_primitives",
        ["cpp_src/rotating_primitives.cpp", "cpp_src/exp.cpp", "cpp_src/rotating_primitives_cuda.cu"],
        extra_compile_args={
            'cxx': [],
            'nvcc': ['-O3', '--use_fast_math']
        }
    ),
]

setup(
    name="rotating_primitives",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
