from setuptools import find_packages, setup

# package name
_name = 'finetuner'

# package long description
try:
    with open('README.md', encoding='utf8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''


if __name__ == '__main__':
    setup(
        name=_name,
        packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
        include_package_data=True,
        description='Task-oriented finetuning for better embeddings on neural search.',
        author='Jina AI',
        author_email='hello@jina.ai',
        url='https://github.com/jina-ai/finetuner/',
        license='Apache 2.0',
        download_url='https://github.com/jina-ai/finetuner/tags',
        long_description=_long_description,
        long_description_content_type='text/markdown',
        zip_safe=False,
        setup_requires=['setuptools>=18.0', 'wheel'],
        install_requires=[
            'docarray[common]>=0.21.0',
            'trimesh==3.16.4',
            'finetuner-stubs==0.12.7',
            'jina-hubble-sdk==0.33.1',
        ],
        extras_require={
            'full': [
                'finetuner-commons==0.12.7',
            ],
            'test': [
                'black==22.3.0',
                'flake8==5.0.4',
                'isort==5.10.1',
                'pytest==7.0.0',
                'pytest-cov==3.0.0',
                'pytest-mock==3.7.0',
            ],
        },
        python_requires='>=3.8.0',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'License :: OSI Approved :: Apache Software License',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        project_urls={
            'Documentation': 'https://finetuner.jina.ai',
            'Source': 'https://github.com/jina-ai/finetuner/',
            'Tracker': 'https://github.com/jina-ai/finetuner/issues',
        },
        keywords=(
            'jina neural-search neural-network deep-learning pretraining '
            'fine-tuning pretrained-models triplet-loss metric-learning '
            'siamese-network few-shot-learning'
        ),
    )
