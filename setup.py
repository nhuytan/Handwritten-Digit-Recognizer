from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='Handwritten-Digit-Recognizer',
    version='1.1',
    author='Brandon Nguyen', 
    author_email='nhuytan@mail.com',
    packages=find_packages(),
    long_description=open('README.md').read()
)
