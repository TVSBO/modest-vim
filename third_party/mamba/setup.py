"""Build the bundled BiMamba-compatible selective-scan extension."""
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
sources = ['selective_scan.cpp', 'selective_scan_fwd_fp32.cu',
           'selective_scan_fwd_fp16.cu', 'selective_scan_fwd_bf16.cu']
sources += [f'selective_scan_bwd_{dtype}_{kind}.cu'
            for dtype in ('fp32', 'fp16', 'bf16') for kind in ('real', 'complex')]
flags = ['-O3', '-std=c++17', '-U__CUDA_NO_HALF_OPERATORS__',
         '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_OPERATORS__',
         '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_BFLOAT162_OPERATORS__',
         '-U__CUDA_NO_BFLOAT162_CONVERSIONS__', '--expt-relaxed-constexpr',
         '--expt-extended-lambda', '--use_fast_math']
setup(name='mamba_ssm', version='1.1.1', packages=find_packages(),
      ext_modules=[CUDAExtension('selective_scan_cuda',
          sources=[str(root / 'csrc' / 'selective_scan' / name) for name in sources],
          include_dirs=[str(root / 'csrc' / 'selective_scan')],
          extra_compile_args={'cxx': ['-O3', '-std=c++17'], 'nvcc': flags})],
      cmdclass={'build_ext': BuildExtension}, python_requires='>=3.10',
      license='Apache-2.0')
