FROM conda/miniconda3

MAINTAINER Maxim Kochurov <maxim.v.kochurov@gmail.com>
ARG SRC_DIR
ARG PYTORCH
ENV PYTORCH=${PYTORCH}
COPY $SRC_DIR/scripts/create_testenv.sh /opt/geoopt/create_testenv.sh
COPY $SRC_DIR/requirements-dev.txt /opt/geoopt/requirements-dev.txt
COPY $SRC_DIR/scripts/create_testenv.sh /opt/geoopt/scripts/create_testenv.sh

WORKDIR /opt/geoopt
RUN ./scripts/create_testenv.sh
COPY $SRC_DIR /opt/geoopt
RUN pip install --no-deps -e .
RUN find -type d -name __pycache__ -exec rm -rf {} +
# Set automatic conda activation in non interactive and shells
