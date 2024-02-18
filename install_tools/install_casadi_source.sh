#!/bin/bash


#refer to https://github.com/casadi/casadi/wiki/InstallationLinux
#NOTE YOU ONLY NEED TO DO THIS ONCE, IF YOU HAVE TO REDO IT AGAIN 
#DELETE YOUR BUILD DIRECTORY AND RERUN THIS SHELL SCRIPT
#change directory to home
cd ~

#download main dependencies for python with casadi
sudo apt install ipython3 python3-dev python3-numpy python3-scipy python3-matplotlib --install-recommends
sudo apt install swig --install-recommends

#clone casadi
# check if casadi is already installed
# check if casadi is installed in the system if not install it
if [ -d "casadi" ]; then
  echo "casadi repo exists"
else
    echo "casadi is not installed"
    git clone https://github.com/casadi/casadi.git -b main casadi
fi

#change directory to casadi
cd casadi
git pull

#mkdir build
#check if build directory exists
if [ -d "build" ]; then
  echo "build directory exists"
  exit 1
else
    echo "build directory does not exist"
    mkdir build
fi
cd build
cmake ../ -DWITH_PYTHON=ON -DWITH_PYTHON3=ON -DWITH_IPOPT=ON -DWITH_OPENMP=ON -DWITH_CLANG=ON

#check how many cores are available
nproc - 1
#compile with all cores
make -j$(nproc)
sudo make install
