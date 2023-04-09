from setuptools import setup, find_packages

setup(
    name='mlxu',
    version='0.1.10',
    license='MIT',
    description='Machine learning experiment utils.',
    url='https://github.com/young-geng/mlxu',
    packages=find_packages(include=['mlxu']),
    author='Xinyang (Young) Geng',
    author_email='young.gengxy@gmail.com',
    python_requires=">=3.7",
    install_requires=[
        'absl-py',
        'ml-collections',
        'wandb',
        'gcsfs',
        'cloudpickle',
        'numpy',
    ],
    extras_require={
        'jax': [
            'jax',
            'flax',
            'optax',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
)
