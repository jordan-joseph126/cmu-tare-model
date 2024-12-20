import sys

if "upload" in sys.argv:
    print("Uploading to PyPI is disabled for this package.")
    sys.exit(1)

from setuptools import setup, find_packages

setup(
    name="cmu_tare_model",
    version="2.0",
    packages=find_packages(),
    install_requires=[
        # Add dependencies here, e.g., 'numpy', 'pandas', etc.
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'seaborn',
    ],
    description="Tradeoff Analysis of Residential retrofits for Energy equity (TARE) Model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jordan Joseph",
    author_email="jordanjo@andrew.cmu.edu",
    url="https://github.com/jordanjoseph/cmu-tare-model",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering"
    ],
    python_requires=">=3.8",  # Updated Python requirement
)