FROM conda/miniconda3

LABEL maintainer="Maxim Kochurov <maxim.v.kochurov@gmail.com>"
# source files for geoopt
ARG SRC_DIR
# pytorch version
ARG PYTORCH=pytorch
# python version
ARG PYTHON_VERSION=3.6
# additional requirements
RUN apt-get update && apt-get install -y git build-essential pandoc vim \
    && rm -rf /var/lib/apt/lists/*
COPY $SRC_DIR/requirements-dev.txt /opt/geoopt/requirements-dev.txt
# install most of the things using conda
# start with python version and numpy with mkl support, then the rest
RUN conda update -n base -c defaults conda && \
    conda install --yes python=${PYTHON_VERSION} pip numpy mkl-service && \
    conda install --yes ${PYTORCH} -c pytorch && \
    pip install -r /opt/geoopt/requirements-dev.txt
# set source dir as working dir
WORKDIR /opt/geoopt
# copy sources from local machine
COPY $SRC_DIR /opt/geoopt
# install the package ignoring deps in setup.py
RUN pip install --no-deps -e .
# local copy version is dirty, clean __pycache__ files to avoid weird errors
RUN find -type d -name __pycache__ -exec rm -rf {} +
