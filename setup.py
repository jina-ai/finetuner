import os

from setuptools import find_packages, setup

# package name
_name = 'finetuner-client'


# package version
__version__ = '0.0.0'
try:
    libinfo_py = os.path.join('finetuner', '__init__.py')
    libinfo_content = open(libinfo_py, 'r', encoding='utf8').readlines()
    version_line = [
        line.strip() for line in libinfo_content if line.startswith('__version__')
    ][0]
    exec(version_line)  # gives __version__
except FileNotFoundError:
    pass


# package metadata
_description = (
    'Finetuner allows one to tune the weights of any deep neural network for '
    'better embeddings on search tasks.'
)
_setup_requires = ['setuptools>=18.0', 'wheel']
_python_requires = '>=3.7.0'
_author = 'Jina AI'
_email = 'hello@jina.ai'
_keywords = (
    'jina neural-search neural-network deep-learning pretraining '
    'fine-tuning pretrained-models triplet-loss metric-learning '
    'siamese-network few-shot-learning'
)
_url = 'https://github.com/jina-ai/finetuner/'
_download_url = 'https://github.com/jina-ai/finetuner/tags'
_classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'License :: OSI Approved :: Apache Software License',
    'Environment :: Console',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]
_project_urls = {
    'Documentation': 'https://finetuner.jina.ai',
    'Source': 'https://github.com/jina-ai/finetuner/',
    'Tracker': 'https://github.com/jina-ai/finetuner/issues',
}
_license = 'Apache 2.0'
_package_exclude = ['*.tests', '*.tests.*', 'tests.*', 'tests']


# package requirements
try:
    with open('requirements.txt', 'r') as f:
        _main_deps = f.readlines()
except FileNotFoundError:
    _main_deps = []


# package long description
try:
    with open('README.md', encoding='utf8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''


if __name__ == '__main__':
    setup(
        name=_name,
        packages=find_packages(exclude=_package_exclude),
        version=__version__,
        include_package_data=True,
        description=_description,
        author=_author,
        author_email=_email,
        url=_url,
        license=_license,
        download_url=_download_url,
        long_description=_long_description,
        long_description_content_type='text/markdown',
        zip_safe=False,
        setup_requires=_setup_requires,
        install_requires=_main_deps,
        python_requires=_python_requires,
        classifiers=_classifiers,
        project_urls=_project_urls,
        keywords=_keywords,
    )
