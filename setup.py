import setuptools
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="fastt5",
    version="0.0.7",
    license="apache-2.0",
    author="Kiran R",
    author_email="kiranr8k@gmail.com",
    description="boost inference speed of T5 models by 5x & reduce the model size by 3x using fastT5.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ki6an/fastT5",
    project_urls={
        "Repo": "https://github.com/Ki6an/fastT5",
        "Bug Tracker": "https://github.com/Ki6an/fastT5/issues",
    },
    keywords=[
        "T5",
        "ONNX",
        "onnxruntime",
        "NLP",
        "transformer",
        "quantization",
        "generate text",
        "summarization",
        "translation",
        "q&a",
        "qg",
        "machine learning",
        "inference",
        "fast inference",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "torch>=1.7.0,!=1.8.0",  # excludes torch v1.8.0
        "onnx",
        "onnxruntime==1.7.0",
        "transformers==4.6.1",
        "progress>=1.5",
        "sentencepiece",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
