# Set conda path info
if [[ "$TRAVIS_OS_NAME" != "windows" ]]; then
    export MINICONDA_PATH=$HOME/miniconda;
    export MINICONDA_SUB_PATH=$MINICONDA_PATH/bin;
elif [[ "$TRAVIS_OS_NAME" == "windows" ]]; then
    export MINICONDA_PATH=$HOME/miniconda;
    export MINICONDA_PATH_WIN=`cygpath --windows $MINICONDA_PATH`;
    export MINICONDA_SUB_PATH=$MINICONDA_PATH/Scripts;
fi;
export MINICONDA_LIB_BIN_PATH=$MINICONDA_PATH/Library/bin;
  # Obtain miniconda installer
if [[ "$TRAVIS_OS_NAME" != "windows" ]]; then
    mkdir -p $HOME/download;
    if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
        echo "downloading miniconda.sh for linux";
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/download/miniconda.sh;
    elif  [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
        echo "downloading miniconda.sh for osx";
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O $HOME/download/miniconda.sh;
    fi;
fi;
  # Install openssl for Windows
if [[ "$TRAVIS_OS_NAME" == "windows" ]]; then
    choco install openssl.light;
fi;