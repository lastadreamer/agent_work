from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "xiangqi_engine._xiangqi",
    ["src/board.cpp", "src/encode.cpp", "src/bindings.cpp"],
    include_dirs=["include"],
    cxx_std=17,
    extra_compile_args=["-O3", "-DNDEBUG", "-fvisibility=hidden"],
)

setup(
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
