from setuptools import setup, find_packages

setup(
    name='SPADE_surrogates',
    version='1.0.0',  # Fixed: Valid semantic version instead of '1.'
    description="Code for the analyses of Stella, Bouss et. al (2022)",
    
    author="Peter Bouss",
    author_email="p.bouss@fz-juelich.de",
    
    license='BSD',
    
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    
    packages=find_packages(),  # Automatically find all packages
    
    python_requires='>=3.8',  # Specify minimum Python version
    
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'quantities>=0.12.0',
        'elephant>=0.11.0',
        'neo>=0.10.0',
        'pyyaml>=5.4.0',
        'mpi4py>=3.1.0',
        'snakemake>=7.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'pandas>=1.3.0',
    ],
    
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'flake8>=3.8',
            'black>=21.0',
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
        ],
        'plotting': [
            'bokeh>=2.4.0',
            'altair>=4.2.0',
            'plotnine>=0.8.0',
            'holoviews>=1.14.0',
            'plotly>=5.0.0',
        ],
        'analysis': [
            'scikit-learn>=1.0.0',
            'statsmodels>=0.13.0',
            'networkx>=2.6.0',
            'h5py>=3.6.0',
            'tqdm>=4.62.0',
        ],
    },
    
    entry_points={
        'console_scripts': [
            # Add any command-line scripts here if needed
            # 'spade-analysis=SPADE_surrogates.cli:main',
        ],
    },
    
    include_package_data=True,
    zip_safe=False,
)