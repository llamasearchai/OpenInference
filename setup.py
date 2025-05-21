from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

# C++ extension module
class BuildExt(build_ext):
    def build_extensions(self):
        # Add necessary compiler flags
        if self.compiler.compiler_type == 'unix':
            for ext in self.extensions:
                ext.extra_compile_args = ['-std=c++20', '-O3', '-Wall', '-Wextra']
        super().build_extensions()

cpp_extension = Extension(
    'openinference.runtime.core._core',
    sources=[
        'OpenInference/runtime/core/src/inference_engine.cpp',
        'OpenInference/runtime/core/src/memory_manager.cpp',
        'OpenInference/runtime/core/src/batching.cpp',
        'OpenInference/runtime/core/src/optimizations.cpp',
    ],
    include_dirs=['OpenInference/runtime/core/include'],
    language='c++',
)

setup(
    name="openinference",
    version="0.1.0",
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="An open framework for optimizing and serving Large Language Models and other ML models.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/llamasearchai/OpenInference",
    keywords=["llm", "inference", "deep learning", "machine learning", "model serving", "optimization", "transformers"],
    packages=find_packages(),
    ext_modules=[cpp_extension],
    cmdclass={'build_ext': BuildExt},
    install_requires=[
        "torch==2.0.1",
        "fastapi==0.103.1",
        "uvicorn==0.23.2",
        "numpy>=1.24.0",
        "transformers>=4.31.0",
        "pydantic>=2.3.0",
        "structlog==23.1.0",
        "tensorrt==8.6.1",
        "triton>=2.0.0",
        "safetensors>=0.3.1",
        "langchain",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "mypy>=1.4.1",
            "sphinx>=7.0.0",
        ],
        "ui": [
            "plotly==5.16.1",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
)