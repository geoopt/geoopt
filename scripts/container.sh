#!/usr/bin/env bash
# prototype is taken from  https://github.com/arviz-devs/arviz/blob/master/scripts/container.sh
SRC_DIR=${SRC_DIR:-`pwd`}
COVERAGE=${COVERAGE:-"--cov geoopt"}
PYTORCH=${PYTORCH:-"pytorch>=1.0.0"}
# Build container for use of testing or notebook
if [[ $* == *--build* ]]; then
    echo "Building Docker Image"
    docker build \
        -t geoopt \
        -f $SRC_DIR/scripts/Dockerfile \
        --build-arg SRC_DIR=. $SRC_DIR \
        --build-arg PYTORCH=${PYTORCH} \
        --rm
fi

if [[ $* == *--test* ]]; then
    echo "Testing Geoopt"
    docker run -it --user $(id -u):$(id -g) --mount type=bind,source="$(pwd)",target=/opt/geoopt/ geoopt:latest bash -c \
                                      "pytest -v tests/ ${COVERAGE}/"
fi

if [[ $* == *--bash* ]]; then
    echo "Running Bash"
    docker run -it --user $(id -u):$(id -g) --mount type=bind,source="$(pwd)",target=/opt/geoopt/ geoopt:latest bash
fi
