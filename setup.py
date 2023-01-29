from setuptools import setup, find_packages

setup(
    name='mlxu',
    version='0.1.0',
    license='MIT License',
    long_description='Machine learning experiment utils.',
    packages=find_packages(include=['mlxu']),
    install_requires=[
        'absl-py',
        'ml-collections',
        'wandb',
        'gcsfs',
        'cloudpickle',
    ]
)