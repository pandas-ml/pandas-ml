if [ "$TRAVIS_PYTHON_VERSION" == "2.7" ]; then
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
else
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi
bash conda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda

#Useful for debugging any issues with conda
conda info -a

conda create -n myenv "python=$TRAVIS_PYTHON_VERSION"
source activate myenv
conda install numpy scipy matplotlib scikit-learn pandas patsy statsmodels nose

python -m pip install python-coveralls coverage
python -m pip install graphviz

# xgboost
git clone https://github.com/dmlc/xgboost.git
cd xgboost
./build.sh
cd python-package
python setup.py install
cd ../..
