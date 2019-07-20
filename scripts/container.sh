#!/usr/bin/env bash
set -e
# prototype is taken from  https://github.com/arviz-devs/arviz/blob/master/scripts/container.sh
SRC_DIR=${SRC_DIR:-`pwd`}
COVERAGE=${COVERAGE:-"--cov geoopt"}
PYTORCH=${PYTORCH:-"pytorch"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.6"}
# Build container for use of testing or notebook
if [[ $* == *--build* ]]; then
    echo "Building Docker Image"
    docker build \
        -t geoopt \
        -f ${SRC_DIR}/scripts/Dockerfile \
        --build-arg SRC_DIR=. ${SRC_DIR} \
        --build-arg PYTORCH=${PYTORCH} \
        --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
        --rm
fi

if [[ $* == *--test* ]]; then
    echo "Testing Geoopt"
    # some workarounds for coverage in docker
    # we want to create a self contained container with library
    # and run tests inside. Docker does allow only inplace file modifications
    # but coverage script overrides the file. Therefore we
    # 1) clean the coverage info
    if [[ ${COVERAGE} ]]; then
        rm -f "$(pwd)/.coverage"
        # 2) create an empty file
        touch "$(pwd)/.coverage"
        # 3) run docker linking the created coverage file to the output file inside the container
        #                      (the created file) -----> (output file)
        docker run --rm -it -v "$(pwd)/.coverage":/opt/geoopt/.coverage_result geoopt:latest \
            bash -c "make lint && \
                     pytest -v geoopt tests --durations=0 --doctest-modules ${COVERAGE} && \
                     cat /opt/geoopt/.coverage >> /opt/geoopt/.coverage_result"
        #                      (coverage info) >> (output file)
        # 4) as usual we run linting
        # 5) but finally we append the generated coverage info to the empty output file
        # 6) paths are wrong so we replace the wrong path with the correct one

        sed -i 's@/opt/geoopt@'${SRC_DIR}'@g' "$(pwd)/.coverage"
    else
        # Run without coverage stuff
        docker run --rm -it geoopt:latest \
            bash -c "make lint && \
                     pytest -v geoopt tests --durations=0 --doctest-modules"
    fi
fi

if [[ $* == *--bash* ]]; then
    echo "Running Bash"
    docker run --rm -it geoopt:latest bash
fi
