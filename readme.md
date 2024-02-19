# Installation 

## To do 
- Install Third-Party HSL from source

## Build Casadi from Source
This repo requires you to build casadi from source, luckily all the commands needed to set up the casadi python package is done with the install_tools/install_casadi_source.sh script. To run simply do the following:

```
cd install_tools #change directory to install tools
chmod +x install_casadi_source.sh #make the shell script executable
./install_casadi_source.sh #run the script
```
Once you are done run the set_casadi_path to set the Casadi python packages to your PYTHON_PATH global environment, do that by doing the following

```
cd install_tools #if you are in the directory yet
./set_casadi_path 
```

## Set up the python virtual environment
If this is the first time you setup your virtual environment do the following command, NOTE YOU ONLY NEED TO DO THIS ONCE
```
activate venv
```

This will set up a virtual environment for your development so your Python Packages don't get installed to your main system, now to activate your virtual environment we have a shell script set up for you, just do the following command. Use this command whenever you want to start your development

```
source activate_venv.sh
```

To install all your Python Packages you need (you only need to do this once), first activate your virtual environment and do the following command
```
pip install -r requirements.txt
```
Now you are good go!


