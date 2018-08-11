git pull

cd fler-utils
git reset --hard
git pull
pip install  -e .
cd ..

cd fler-core
git reset --hard
git pull
pip install -e .
cd ..

cd fler-api
git reset --hard
git pull
pip install -e .
cd ..