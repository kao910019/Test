# install python in one command
# make tmp dir
mkdir tmp
cd tmp
# download python
# you can change your version, check https://www.python.org/ftp/python
wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tgz
# unzip
tar -xvf Python-3.6.3.tgz
# run the configure script
cd Python-3.6.3
# configure install place
./configure --prefix=$HOME/.local/python
make && make install
# Done!
if grep -q "alias python '~/.local/python/bin/python3.6'" ~/.cshrc;
then
  echo "# Install Complete."
else
  echo "" >> ~/.cshrc
  echo "# For python command" >> ~/.cshrc
  echo "alias python '~/.local/python/bin/python3.6'" >> ~/.cshrc
  echo "alias pip '~/.local/python/bin/pip3.6'" >> ~/.cshrc
  echo "alias tensorboard 'python ~/.local/python/lib/python3.6/site-packages/tensorboard/main.py'" >> ~/.cshrc
fi

