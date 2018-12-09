from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='geoopt',
        author='Maxim Kochurov',
        packages=find_packages(),
        install_requires=['torch>=0.4.1', 'numpy']
    )
