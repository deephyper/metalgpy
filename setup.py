from setuptools import setup, find_packages

# What packages are required for this module to be executed?
REQUIRED = [
    "dm-tree",
    "numpy",
]



setup(name="metalgpy", packages=find_packages(), install_requires=REQUIRED)
