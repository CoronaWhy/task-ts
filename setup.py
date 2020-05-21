import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
with open('requirements.txt') as f:
    required = f.read().splitlines()
    
setuptools.setup(
    name="corona-ts-tools", # Replace with your own username
    version="0.0.1",
    author="Example Author",
    author_email="igodfried@isaac26.com",
    description="A package for forecasting CoronaVirus infection rates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=required
)
