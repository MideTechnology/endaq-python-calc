import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    "numpy>=1.18",
    "pandas>=1.2",
    "scipy>=1.6",
    ]

TEST_REQUIRES = [
    "pytest",
    "pytest-cov",
    "hypothesis",
    "sympy",
    ]

EXAMPLE_REQUIRES = [
    ]

setuptools.setup(
        name='endaq-calc',
        version='1.1.0.post1',
        author='Mide Technology',
        author_email='help@mide.com',
        description='a computational backend for vibration analysis',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/MideTechnology/endaq-python-calc',
        license='MIT',
        classifiers=['Development Status :: 4 - Beta',
                     'License :: OSI Approved :: MIT License',
                     'Natural Language :: English',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'Topic :: Scientific/Engineering',
                     ],
        keywords='ebml binary ide mide endaq',
        packages=['endaq.calc'],
        package_dir={'endaq.calc': './endaq/calc'},
        install_requires=INSTALL_REQUIRES,
        extras_require={
            'test': INSTALL_REQUIRES + TEST_REQUIRES,
            'example': INSTALL_REQUIRES + EXAMPLE_REQUIRES,
            },
)
