from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="csr_cu_extension",
    ext_modules=[
        CUDAExtension(
            "csr_cusparse",
            ["csr.cu"],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
