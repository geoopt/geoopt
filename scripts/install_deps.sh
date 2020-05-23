conda install --yes pip numpy mkl-service
conda install --yes pytorch${PYTORCH_VERSION} cpuonly -c ${PYTORCH_CHANNEL}
pip install -r ./requirements-dev.txt
