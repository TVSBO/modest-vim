"""Build causal-conv1d 1.4.0 for the local CUDA architecture."""
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
sources = ['causal_conv1d.cpp', 'causal_conv1d_fwd.cu',
           'causal_conv1d_bwd.cu', 'causal_conv1d_update.cu']
flags = ['-O3', '-std=c++17', '-U__CUDA_NO_HALF_OPERATORS__',
         '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_OPERATORS__',
         '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_BFLOAT162_OPERATORS__',
         '-U__CUDA_NO_BFLOAT162_CONVERSIONS__', '--expt-relaxed-constexpr',
         '--expt-extended-lambda', '--use_fast_math']
setup(name='causal_conv1d', version='1.4.0', packages=find_packages(),
      ext_modules=[CUDAExtension('causal_conv1d_cuda',
          sources=[str(root / 'csrc' / name) for name in sources],
          include_dirs=[str(root / 'csrc')],
          extra_compile_args={'cxx': ['-O3', '-std=c++17'], 'nvcc': flags})],
      cmdclass={'build_ext': BuildExtension}, python_requires='>=3.10',
      license='BSD-3-Clause')
