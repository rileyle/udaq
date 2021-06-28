import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="udaq",
    version="0.1",
    author="Lew Riley",
    author_email="lriley@ursinus.edu",
    description="DAQ package for PicoScope 5000-Series PC oscilloscopes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rileyle/udaq",
    packages=setuptools.find_packages(),
#    package_data={
#        '': ,
#    },
    entry_points={
        'console_scripts': [
            'udaq=udaq.udaq:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
)
