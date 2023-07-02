---
layout: post
title:  Installation Updates of Sim-to-Real-Virtual-Guidance-for-Robot-Navigation
date: 2023-07-02 00:00:00-0400
description: Additional installation steps for old repo.
categories: tutorial
tags: 
disqus_comments: true
related_posts: false
toc:
  sidebar: left
---

The article provides clear instructions on how to install necessary environments for all modules in <a href="https://github.com/KaiChen1008/Sim-to-Real-Virtual-Guidance-for-Robot-Navigation">Sim-To-Real Navigation Robots</a> project, apart from the official <a href="https://kaichen1008.github.io/Sim-to-Real-Virtual-Guidance-for-Robot-Navigation/">original documentation</a> provided, since it was made in 2020, some additional corrections are needed during the installation.

Moreover, there are some missing files in the original repo, they were also provided here. Note that this article will only go though the updates to the original documentation, please refer to the original documentation for full instructions.


## ORB-SLAM2 Installation

---


### OpenCV 3.2

**1. Missing stdlib.h**

In section 3.3, change the original build & install command into the following:

```
cmake -DCMAKE_BUILD_TYPE=Release \
      -DWITH_CUDA=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-${CV3_RELEASE} \
      -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
      /tmp/opencv
      -DENABLE_PRECOMPILED_HEADERS=OFF
```

`-DENABLE_PRECOMPILED_HEADERS=OFF` is added, without this line of parameter might result in error of missing _stdlib.h_ (fatal error: stdlib.h: No such file or directory #include_next stdlib.h).

**2. Specify Number of Parallel Building Threads**

If you are running Ubuntu 18.04 in virtual machine, it is likely that ``proc`` variable does not work, hence leading to an building error.

Make OpenCV by explicitly specify the number of threads would address this problem, this is also recommended:

```
sudo make install -j8
```

In this case 8 threads will be used to build the project.


### Pangolin

Do not just clone the Pangolin repo directly, Pangolin has several upgrades to the new version, which were made compilable with C++ 17. However, C++ 17 is not installed in Ubuntu 18.04 by default, install latest version of Pangolin in lower version C++ will cause error [Pangolin could not be found because dependency Eigen3 could not be found.].

Download Pangolin v0.5 <a href="https://github.com/stevenlovegrove/Pangolin/tags">here</a> and build & install it as detailed in original instruction.

Make sure you specify threads while building Pangolin if needed.

### Install ORB-SLAM2

Clone the repo to your local environment first as instructed by the original documentation. Copy _modified_ORB_SLAM2_ folder in _localization_module_ folder to the `src` directory of your ROS workspace, and rename it to _ORB_SLAM2_. 

In additional, you must do the following first **before** build ORB-SLAM2.

**1. Missing tar file**

In _built.bash_, file _ORBvoc.txt.tar.gz_ is referenced but not provided. Download the file <a href="https://github.com/raulmur/ORB_SLAM2/blob/master/Vocabulary/ORBvoc.txt.tar.gz">here</a> and put it into _ORB-SLAM2/Vocabulary/_.

**2. Missing Examples Folder**

The original __Examples__ folder provided is not complete. Replace the whole folder with the one appears in <a href="https://github.com/raulmur/ORB_SLAM2/tree/master/Examples">this repo.</a> 

**3. Update CMakeList.txt**

Update _CMakeList.txt_ in _ORB_SLAM2_Examples/ROS/ORB_SLAM2/_ by adding a boost library line `-lboost_syste` into the library section:

```
set(LIBS
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../../Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../../Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../../lib/libORB_SLAM2.so
-lboost_system
)

```

Now you can build & install ORB-SLAM2 and ORB-SLAM2 node as instructed by the original documentation.


