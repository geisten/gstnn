<h1 align="center">
   <img src="./doc/img/neuron.png" alt="geisten neurons">
</h1>
<h4 align="center">The minimal c deep learning api you are looking for</h4>

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Platforms](https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20OS%20%7C%20BSD-blue.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/geisten/geisten.svg)](https://github.com/geisten/geisten/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/geisten/geisten.svg)](https://github.com/geisten/geisten/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

---

# About

gstnn is a neural net processing engine written in C. It allows you to create your own neural net. geisten is not meant
for newbies and non-programmers. The library is designed for “scalability, speed, and ease-of-use” and optimized for
Unix (*BSD, Linux, MacOSX) like environments.

## Getting Started

Create a source directory and download the source from github:

```shell
git clone https://github.com/geisten/gstnn
```

Change to the directory to `cd gstnn/data` and download the mnist training and testing data

```shell
sh mnist_data_download.sh
```

Change to the directory to `cd gstnn` and run `make mnist` to build the mnist the geisten program.

```shell
make mnist
```

Run the mnist neural net program

```shell
.\gstnn -t ./data/mnist_targets_train.f32 ./data/mnist_images_train.f32
```

The tests will take a while to execute.

make install

To execute all unit tests - run the following program within the build directory:

```shell
make tests
```

The tests will take a while to execute.

## Motivation

gstnn is a minimalistic neural network written in C.

## Differences

In contrast to Keras, Tensorflow, etc. gstnn is much smaller and simpler.

- strongly reduced functionality compared to the aforementioned alternatives
- gstnn has no Lua integration, no shell-based configuration and comes without any additional tools.
- gstnn is only a single binary, and the source code of the reference implementation is intended to never exceed 1000
  SLOC.
- gstnn is customized through editing its source code, which makes it extremely fast and secure - it does not process
  any input data which isn't known at compile time. You don't have to activate Lua/Python or some weird configuration
  file format, beside C, to customize it for your needs: you only have to activate C (at least in order to edit the
  header file)
  .
- Because gstnn is customized through editing its source code, it's pointless to make binary packages of it.

## Configuration

Configuration is done with config.h. It can be edited just like any other C source code file. It contains definitions of
variables that are going to be used by gstnn.c and therefore it is vital that the file is always up to date. Read the
comments in the generated config.h to edit it according to your needs.

## Support

See the faq for the frequent problems that arise. The next step is to look at the sourcecode and the config.h for
obvious names, which could be related to the problem that arose. If that does not help to fix the problem, then there is
the #suckless IRC channel and the mailing list.

If it is your first time using dwm, start with reading the tutorial.

## Development

gstnn is actively developed. You can browse its source code repository or get a copy using git with the following
command:

    git clone https://github.com/geisten/gstnn.git

