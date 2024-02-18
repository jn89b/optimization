#cd into root 
cd ~

cd casadi/build/lib

#set a variable to the path
casadi_path=$(pwd)

#set to pythonpath
#check if already in bashrc
echo "export PYTHONPATH=$casadi_path:$PYTHONPATH" >> ~/.bashrc

#print out the path
echo "set the python path to $casadi_path"
