from setuptools import setup, find_packages

setup(
    name='pyjobs',
    version='0.1.0',
    author='Matteo Saccardi',
    author_email='matteo.saccardi97@gmail.com',
    description='Jackknife-based observable statistics and error propagation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MatteoSaccardi/pyjobs',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: GNU GPLv2",
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'matplotlib',
        'sympy'
    ],
)