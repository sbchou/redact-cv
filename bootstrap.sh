sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y python
sudo apt-get install -y python-pip
sudo apt-get install -y vim
sudo apt-get install git-core
#numpy stack
sudo apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
#opencv
sudo apt-get install -y build-essential libgtk2.0-dev libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen2-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev
wget http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.6.1/opencv-2.4.6.1.tar.gz
tar xvfz opencv-2.4.6.1.tar.gz
cd opencv-2.4.6.1
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_PYTHON_SUPPORT=ON -D BUILD_EXAMPLES=ON ..
#cmake -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON ..
make
sudo make install -y
sudo echo '/usr/local/lib' >> /etc/ld.so.conf.d/opencv.conf
sudo ldconfig
sudo echo 'PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH' >> /etc/bash.bashrc
sudo pip install -y  --upgrade ipython
sudo apt-get install -y chromium-browser
sudo apt-get install -y  build-essential python-dev python-numpy python-setuptools python-scipy libatlas-dev libatlas3-base
sudo apt-get install -y python-sklearn
sudo apt-get install -y unpaper
sudo apt-get install -y imagemagick
sudo apt-get install -y r-base-core
sudo apt-get install -y git-core

sudo reboot

echo 'Meow meow'
