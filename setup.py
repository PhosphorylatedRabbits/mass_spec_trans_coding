"""Install package."""
from setuptools import setup, find_packages

setup(
    name='mstc',
    version='0.2',
    description='Transferred encoding of mass spectrometry images.',
    long_description=open('README.md').read(),
    url='https://github.com/PhosphorylatedRabbits/mass_spec_trans_coding',
    author='Joris Cadow',
    author_email='joriscadow@gmail.com',
    packages=find_packages('.'),
    install_requires=[
        "numpy",
        "scipy",
        "imageio",
        "pandas",
        "scikit-learn",
        "scikit-image",
        "dask[array]",
        "xarray",
        "tensorflow",
        "tensorflow_hub",
    ],
    zip_safe=False,
)
