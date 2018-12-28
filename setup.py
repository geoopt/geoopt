import os
import re
from setuptools import setup, find_packages


def get_version(*path):
    version_file = os.path.join(*path)
    lines = open(version_file, "rt").readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError("Unable to find version in %s." % (version_file,))


if __name__ == "__main__":
    setup(
        name="geoopt",
        author="Maxim Kochurov, Victor Yanush",
        packages=find_packages(),
        install_requires=["torch>=0.4.1", "numpy"],
        version=get_version("geoopt", "__init__.py"),
        url="https://github.com/ferrine/geoopt",
    )
