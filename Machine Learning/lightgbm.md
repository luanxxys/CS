# LightGBM

### Installation
> GUN/Linux

安装 lightgbm

    git clone --recursive https://github.com/Microsoft/LightGBM
    cd LightGBM
    mkdir build
    cd build
    cmake ..
    make -j4

安装 python package

    cd ../python-package
    python setup.py install

测试安装结果

    cd examples/python-guide
    python simple_example.py

[reference](https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst)
