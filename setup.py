import sys
from os import path

from setuptools import find_packages
from setuptools import setup

if sys.version_info < (3, 7, 0):
    raise OSError(f'Finetuner requires Python >=3.7, but yours is {sys.version}')

try:
    pkg_name = 'finetuner'
    libinfo_py = path.join(pkg_name, '__init__.py')
    libinfo_content = open(libinfo_py, 'r', encoding='utf8').readlines()
    version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][
        0
    ]
    exec(version_line)  # gives __version__
except FileNotFoundError:
    __version__ = '0.0.0'

try:
    with open('README.md', encoding='utf8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

setup(
    name=pkg_name,
    packages=find_packages(),
    version=__version__,
    include_package_data=True,
    description='Finetuner allows one to tune the weights of any deep neural network for better embedding on search tasks.',
    author='Jina AI',
    author_email='hello@jina.ai',
    license='Apache 2.0',
    url='https://github.com/jina-ai/finetuner/',
    download_url='https://github.com/jina-ai/finetuner/tags',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    setup_requires=['setuptools>=18.0', 'wheel'],
    install_requires=['jina>=2.4.9', 'matplotlib'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    project_urls={
        'Documentation': 'https://finetuner.jina.ai',
        'Source': 'https://github.com/jina-ai/finetuner/',
        'Tracker': 'https://github.com/jina-ai/finetuner/issues',
    },
    keywords='jina neural-search neural-network deep-learning pretraining '
    'fine-tuning pretrained-models triplet-loss paddlepaddle metric-learning labeling-tool siamese-network '
    'few-shot-learning',
)
