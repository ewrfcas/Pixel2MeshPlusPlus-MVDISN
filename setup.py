try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
#pykdtree = Extension(
#    'im2mesh.utils.libkdtree.pykdtree.kdtree',
#    sources=[
#        'im2mesh/utils/libkdtree/pykdtree/kdtree.c',
#        'im2mesh/utils/libkdtree/pykdtree/_kdtree_core.c'
#    ],
#    language='c',
#    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
#    extra_link_args=['-lgomp'],
#)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'utils.libmcubes.mcubes',
    sources=[
        'utils/libmcubes/mcubes.pyx',
        'utils/libmcubes/pywrapper.cpp',
        'utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'utils.libmesh.triangle_hash',
    sources=[
        'utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'utils.libmise.mise',
    sources=[
        'utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'utils.libsimplify.simplify_mesh',
    sources=[
        'utils/libsimplify/simplify_mesh.pyx'
    ]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'utils.libvoxelize.voxelize',
    sources=[
        'utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)


# Gather all extension modules
ext_modules = [
    #pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
    #dmc_pred2mesh_module,
    #dmc_cuda_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
