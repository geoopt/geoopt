# begin installing miniconda
if [[ "$TRAVIS_OS_NAME" != "windows" ]]; then
    echo "installing miniconda for posix";
    bash $HOME/download/miniconda.sh -b -u -p $MINICONDA_PATH;
elif  [[ "$TRAVIS_OS_NAME" == "windows" ]]; then
    echo "folder $MINICONDA_SUB_PATH does not exist"
    echo "installing miniconda for windows";
    choco install miniconda3 --params="'/JustMe /AddToPath:1 /D:$MINICONDA_PATH_WIN'";
fi;
# end installing miniconda

export PATH="$MINICONDA_PATH:$MINICONDA_SUB_PATH:$MINICONDA_LIB_BIN_PATH:$PATH";

# begin checking miniconda existance
echo "checking if folder $MINICONDA_SUB_PATH exists"
if [[ -d $MINICONDA_SUB_PATH ]]; then
    echo "folder $MINICONDA_SUB_PATH exists"
else
    echo "folder $MINICONDA_SUB_PATH does not exist"
fi;
# end checking miniconda existance

source $MINICONDA_PATH/etc/profile.d/conda.sh;
hash -r;
echo $TRAVIS_OS_NAME
echo "updating conda"
python --version
conda config --set always_yes yes --set changeps1 no;
conda update -q conda;
# Useful for debugging any issues with conda
conda info -a

# See if test-environment already available
# As necessary, apply python module recipies
echo "create test-environment";
echo "Going to install python $PYTHON_VERSION"
conda create -n test-environment python=${PYTHON_VERSION};

conda activate test-environment

conda list
# end python 3.7.4 h265db76_0