import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FLER-utils",
    version="0.0.1",
    author="Axel Marchand",
    author_email="axel-marchand@hotmail.fr",
    description="Utility package for FLER",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Axeldnahcram/FLER",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)