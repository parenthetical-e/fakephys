from setuptools import setup

setup(
    name='fakephys',
    version='0.0.1',
    description='A library for simulating electrophysiological features.',
    url='https://github.com/brainprosthesis/fakephys',
    author='Erik J Peterson',
    author_email='erik.peterson@kernel.co',
    license='MIT',
    packages=['fakephys'],
    scripts=['bin/phys'],
    zip_safe=False)
