import setuptools

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name='zombies',
    version='0.1.0',
    description='Bootstrapping statistics.',
    long_description=long_description,
    long_description_context_type='text/markdown',
    author='Charles D. Holmes',
    author_email='holmes@wustl.edu',
    url='https://github.com/holmescharles/zombies',
    py_modules=['zombies'],
    install_requires=[
        'joblib',
        'numpy',
        'pandas',
        'scipy',
        'tqdm',
        ],
    python_requires='~=3.5',
    )
