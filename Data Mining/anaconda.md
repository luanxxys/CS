# Anaconda

### Install

    https://docs.anaconda.com/anaconda/install/linux

### 管理 conda

查看 conda 版本/检查已安装

    conda --version

升级当前版本的 conda

    conda update conda

删除 conda

    rm -rf ~/miniconda OR  rm -rf ~/anaconda
   > 移除 Anaconda 或 Miniconda 安装文件夹

### 管理环境

创建并激活一个环境

    conda creat -n snowflake biopython
   > 给Biopython创建一个新的环境，位置在/envs/snowflakes

    source activate snowflakes
   > 激活这个环境
   >
   > 未指定 python 版本，则默认为安装 conda 时的版本

创建第二个环境

> 创建并命名一个新环境，然后安装另一个版本的 python 以及两个包 Astroid 和 Babel

    conda create -n bunnies python=3 Astroid Babel
   > 在/envs/bunnies 文件夹
   >
   > 在你创建环境的同时安装好所有你想要的程序，在后来依次安装可能会导致依赖性问题

列出所有的环境/确认当前环境

    conda info --envs

切换到另一个环境(activate/deactivate)

    source activate snowflakes

如果要从你当前工作环境的路径切换到系统根目录时，键入：

    source deactivate

复制一个环境

    conda create -n flowers --clone snowflakes

删除一个环境

    conda remove -n flowers --all

为了确定这个名为flowers的环境已经被移除，输入以下命令

    conda info -e

### 管理 python

检查什么版本的python可以被安装：

    conda search --full --name python

安装一个不同版本的python
> 现在我们假设你需要python3来编译程序，但是你不想覆盖掉你的python2.7来升级，你可以创建并激活一个名为snakes的环境，并通过下面的命令来安装最新版本的python3

    conda create -n snakes python=３

    source activate snakes

确定环境添加成功

    conda info -e

检查新的环境中的python版本

    python --version

注销该环境
> 完成了在snowflakes环境中的工作室，注销掉该环境并转换你的路径到先前的状态

    source deactivate

### 管理包

查看该环境中包和其版本的列表

    conda list

查找一个包
    
    conda search <package_name>

安装一个新包

    conda install --name bunnies <package_name>
   > 你必须告诉 conda 你要安装环境的名字（-n bunies）否则它将会被安装到当前环境中
   >
   > 然后激活 bunnies 环境，并且用 'conda list' 来显示哪些程序被安装了

从 Anaconda.org 安装一个包
> 如果一个包不能使用 conda 安装，我们接下来将在 Anaconda.org 网站查找

    conda install --channel https：//conda .anaconda.ort/pandas bottleneck

通过 pip 命令来安装包
> 对于那些无法通过 conda 安装或者从 Anaconda.org 获得的包，我们通常可以用 pip（“pip install packages”）来安装包

    source activate bunnies
    pip install see
   >  pip 只是一个包管理器，所以它不能为你管理环境,所以首先激活我们想放置程序的环境

[reference:Anaconda入门使用指南](http://www.jianshu.com/p/169403f7e40c)













