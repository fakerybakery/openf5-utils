"""
openf5-utils
Copyright (c) 2025 mrfakename. All rights reserved.
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="openf5-utils",
    description="Utilities for OpenF5 TTS",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="mrfakename",
    author_email="me@mrfake.name",
    version="0.3",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "f5-tts": ["f5-tts"],
        "infer": ["click", "f5-tts", "huggingface-hub", "hf-transfer"],
    },
    url="https://github.com/fakerybakery/openf5-utils",
)
