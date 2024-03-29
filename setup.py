import setuptools

setuptools.setup(
    name="FLutils",                     # This is the name of the package
    version="0.0.9",                        # The initial release version
    author="Davide Domini",                     # Full name of the author
    description="Test how to create a python package",
    long_description="Test how to create a python package",      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["FLutils"],             # Name of the python package
    package_dir={'':'src/'},     # Directory of the source code of the package
    install_requires=["torch"]                     # Install other dependencies if any
)