from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT_DIR = Path(__file__).resolve().parent
COMMON_SOURCES = [
    'geotransformer/extensions/extra/cloud/cloud.cpp',
    'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
    'geotransformer/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
    'geotransformer/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
    'geotransformer/extensions/pybind.cpp',
]


setup(
    name='mfhe',
    version='1.0.0',
    packages=find_packages(
        include=[
            'geotransformer',
            'geotransformer.*',
            'pareconv',
            'pareconv.*',
            'pareGeo',
            'pareGeo.*',
        ]
    ),
    package_data={'geotransformer.modules.kpconv': ['dispositions/*.ply']},
    ext_modules=[
        CUDAExtension(
            name='geotransformer.ext',
            sources=COMMON_SOURCES + [
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
            ],
            include_dirs=[str(ROOT_DIR)],
        ),
        CUDAExtension(
            name='pareconv.ext',
            sources=COMMON_SOURCES + [
                'pareconv/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
            ],
            include_dirs=[str(ROOT_DIR)],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
