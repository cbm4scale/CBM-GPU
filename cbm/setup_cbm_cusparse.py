from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="cbm_cu_extension",
    ext_modules=[
        CUDAExtension(
            "cbm_cusparse",
            ["cbm.cu"],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
