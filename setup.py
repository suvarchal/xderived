# /home/ubuntu/xderived_project/setup.py

from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="xderived", # Updated name
    version="0.3.0", # Updated version to match __init__.py
    author="Suvarchal K. Cheedela",
    author_email="suvarchal@duck.com", # User can change this
    description="An xarray plugin to dynamically register and lazily compute derived scientific variables.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/xderived", # User should update this
    packages=find_packages(include=["xderived", "xderived.*"]), # Updated package find
    install_requires=[
        "xarray>=0.20", # Specify a reasonable minimum version, perhaps newer for latest features
        "numpy>=1.20",
        "jinja2>=3.0", # Added for HTML repr templating
        # Add other dependencies if any, e.g., pint if unit handling is enhanced
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License", # Defaulting to MIT, user can change
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12", # Added Python 3.12 support
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Framework :: Xarray",
    ],
    python_requires=">=3.8", # Updated minimum Python version
    keywords="xarray plugin derived variables meteorology oceanography climate science data analysis xderived",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/xderived/issues", # User should update
        "Source Code": "https://github.com/yourusername/xderived", # User should update
    },
    include_package_data=True, # If you have non-code files in your package
)

