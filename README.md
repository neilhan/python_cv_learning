python_cv_learning
==================

The exercises and learning experience with opencv, and python.

Before the examples can be run, PYTHONPATH needs to be set. 
Excute this command:
> export PYTHONPATH=$$PYTHONPATH:/path_to_python_cv_learning/mylib

To run each example:
Change current directory to each sub directories, type:
> python run_me.py

----------------------------------------------------------------
How to setup openCV on Fedora 19:

following steps are copied from http://aspratyush.wordpress.com/tag/opencv-fedora/

sudo yum groupinstall "Development Tools" "Development Libraries"

sudo yum install eigen2-devel CTL CTL-devel OpenEXR_CTL-libs tbb yasm yasm-devel tbb-devel OpenEXR CTL-devel  gstreamer-plugins-base-devel libsigc++20-devel glibmm24-devel libxml++-devel gstreamermm xine-lib libunicapgtk-devel xine-lib-devel gstreamermm-devel python-devel sip-macros sip vamp-plugin-sdk audacity sip-devel xorg-x11-proto-devel libXau-devel libX11-devel automake libogg-devel libtheora-devel libvorbis-devel libdc1394-devel x264-devel faac-devel xvidcore-devel dirac-devel gsm-devel zlib-devel faad2-devel speex-devel lame-devel orc-compiler orc-devel libvdpau cppunit libvdpau-devel schroedinger-devel dirac x264 lame faad2 amrwb-devel opencore-amr-devel amrnb amrnb-devel

sudo yum install ffmpeg ffmpeg-devel opencv opencv-devel opencv-python

sudo updatedb

sudo yum install ipython ipdb python-opencv python-opengl python-setuptools python-numpy python-scipy python-matplotlib

ipdb install::: easy_install ipdb

Python openCV
----------------------------------------------------------------
How to setup openCV on Ubuntu:

sudo apt-get install python-opencv python-opengl python-setuptools python-numpy python-scipy python-matplotlib
