SZTE_MSc_L industrial image processing project

<h3>Recommended environment</h3>

The project should be run on one of the latest popular gnu/linux distributions, preferably Ubuntu.

<h3>Environment packages</h3>

Running the project will require the python virtualenv package and a python3.8.x installation.
To do that, simply install the virtualenv package available on pypi.org and download a tarball with an older python3.8 release from the official site.

It will also require libbz2-dev libraries, so the interpreter can be compiled with bz2 support and it will also require liblzma-dev.
`apt-get install libbz2-dev liblzma-dev`

<h3>Build python interpreter from source</h3>

Extract the tar archive downloaded in the previous step.
Run the configure script available in the directory where it was extracted. 
Running the configuration script with the option `--prefix=${installation_path}` will define the path where the interpreter should be installed after building. Set it to a directory which the user have access to, since the build may fail when trying to install the interpreter to /usr/local/bin. It is NOT recommended to run the build as root.

Once the build environment is initialised, run `make altinstall`. This command will build the interpreter and install it to the directory defined by `${installation_path}` and will do that without modifying already existing installations.

<h3>Setup python environment</h3>

Once we have the required python installation, we should create a virtualenv with the `--pyton` switch set to the path of the python3.8.x interpreter binary available at `${installation_path}/bin/python3.8`.
After the virtualenv is setup to use the appropriate interpreter version, source the environment and install the dependencies in `requirements.txt`.
