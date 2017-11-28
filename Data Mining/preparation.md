# Data mining preparation

- 代理上网

    + 实验室有境外服务器，或自己有 SS 账号

        [使用 SwitchyOmega 代理上网](https://github.com/luanxxys/software/blob/master/google-chrome/SwitchyOmega.md)

    + 更改 hosts（实验室有线网或不可用）

        * Windows 环境

            找到 `C:\Windows\System32\drivers\etc` 目录下的 `hosts` 文件，以文本方式打开。

            打开下面网址，复制粘贴其中全部内容到 `hosts` 文件中

                https://raw.githubusercontent.com/lennylxx/ipv6-hosts/master/hosts

            重新登陆网络（学校无线网可用）即可。


        * Unix/Linux 环境

            hosts 文件位置：`/etc/hosts`

            其它设置类似 windows 环境


        > [hosts 文件来源](https://github.com/lennylxx/ipv6-hosts)
        >
        > [修改 hosts 文件的原理](https://www.zhihu.com/question/19782572)

- 工具软件

    + github

            代码仓库

    + pycharm

    + onenote

            文档整理、分享交流

    + xmind

            脑图软件

    + RealVNC

            远程桌面软件

    + WinSCP

            和服务器之间文件交互

        > 本地主机是 Unix/Linux 环境下，使用 [SCP](https://github.com/luanxxys/linux/blob/master/memo/scp.md) 命令

    + chrome extensions

            Google Translate
        > 我常用的其它 [chrome 扩展](https://github.com/luanxxys/software/tree/master/google-chrome)

+ Python 环境

    [使用 anaconda 管理 python 虚拟环境](https://github.com/luanxxys/computer-science/blob/master/Data%20Mining/anaconda.md)

+ python 库

    [常用 python 库](https://github.com/luanxxys/computer-science/blob/master/Data%20Mining/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%20-%20Python%20%E5%BA%93.md)

    [pandas](https://github.com/luanxxys/computer-science/blob/master/Data%20Mining/pandas.md)

+ python modules

    [XgBoost](https://github.com/luanxxys/computer-science/blob/master/Data%20Mining/XgBoost.md)

    [LightGBM](https://github.com/luanxxys/computer-science/blob/master/Data%20Mining/LightGBM.md)

+ 服务器操作

    对于一些吃硬件资源的程序，可能需要在服务器上跑

    [初学 Linux 技巧](https://github.com/luanxxys/linux/blob/master/memo/%E5%88%9D%E5%AD%A6%20Linux%20%E6%8A%80%E5%B7%A7.pdf)





