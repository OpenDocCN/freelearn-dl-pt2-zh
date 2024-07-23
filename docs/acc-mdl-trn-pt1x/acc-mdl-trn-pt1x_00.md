# 前言

你好！我是一名专注于**高性能计算**（**HPC**）的系统分析师和学术教授。是的，你没看错！我不是数据科学家。那么，你可能会想知道我为什么决定写一本关于机器学习的书。别担心，我会解释的。

HPC 系统由强大的计算资源紧密集成，用于解决复杂问题。HPC 的主要目标是利用资源、技术和方法加速高强度计算任务的执行。传统上，HPC 环境已被用于执行来自生物学、物理学、化学等多个领域的科学应用程序。

但在过去几年中，情况发生了变化。如今，HPC 系统不仅仅运行科学应用程序的任务。事实上，在 HPC 环境中执行的最显著的非科学工作负载恰恰是本书的主题：复杂神经网络模型的构建过程。

作为数据科学家，您比任何人都知道训练复杂模型可能需要多长时间，以及需要多少次重新训练模型以评估不同场景。因此，使用 HPC 系统加速人工智能（AI）应用程序（不仅用于训练还用于推断）是一个需求增长的领域。

AI 与 HPC 之间的密切关系引发了我对深入研究机器学习和 AI 领域的兴趣。通过这样做，我能更好地理解 HPC 如何应用于加速这些应用程序。

所以，在这里我们是。我写这本书是为了分享我在这个主题上学到的东西。我的使命是通过使用单个或多个计算资源，为您提供训练模型更快的必要知识，并采用优化技术和方法。

通过加速训练过程，你可以专注于真正重要的事情：构建令人惊叹的模型！

# 本书适合谁

本书适合中级数据科学家、工程师和开发人员，他们希望了解如何使用 PyTorch 加速他们的机器学习模型的训练过程。尽管他们不是本材料的主要受众，负责管理和提供 AI 工作负载基础设施的系统分析师也会在本书中找到有价值的信息。

要充分利用本材料，需要具备机器学习、PyTorch 和 Python 的基础知识。然而，并不要求具备分布式计算、加速器或多核处理器的先前理解。

# 本书内容涵盖了什么

*第一章*，*分解训练过程*，提供了训练过程在底层如何工作的概述，描述了训练算法并涵盖了该过程执行的阶段。本章还解释了超参数、操作和神经网络参数等因素如何影响训练过程的计算负担。

*第二章*，*加速训练模型*，提供了加速训练过程可能的方法概述。本章讨论了如何修改软件堆栈的应用和环境层以减少训练时间。此外，它还解释了通过增加资源数量来提高性能的垂直和水平可伸缩性作为另一选项。

*第三章*，*编译模型*，提供了 PyTorch 2.0 引入的新型编译 API 的概述。本章涵盖了急切模式和图模式之间的区别，并描述了如何使用编译 API 加速模型构建过程。此外，本章还解释了编译工作流程及涉及编译过程的各个组件。

*第四章*，*使用专用库*，提供了 PyTorch 用于执行专门任务的库的概述。本章描述了如何安装和配置 OpenMP 来处理多线程和 IPEX 以优化在 Intel CPU 上的训练过程。

*第五章*，*构建高效数据管道*，提供了如何构建高效数据管道以使 GPU 尽可能长时间工作的概述。除了解释数据管道上执行的步骤外，本章还描述了如何通过优化 GPU 数据传输并增加数据管道中的工作进程数来加速数据加载过程。

*第六章*，*简化模型*，提供了如何通过减少神经网络参数的数量来简化模型而不牺牲模型质量的概述。本章描述了用于减少模型复杂性的技术，如模型修剪和压缩，并解释了如何使用 Microsoft NNI 工具包轻松简化模型。

*第七章*，*采用混合精度*，提供了如何采用混合精度策略来加速模型训练过程而不影响模型准确性的概述。本章简要解释了计算机系统中的数值表示，并描述了如何使用 PyTorch 的自动混合精度方法。

*第八章*，*一瞥分布式训练*，提供了分布式训练基本概念的概述。本章介绍了最常用的并行策略，并描述了在 PyTorch 上实施分布式训练的基本工作流程。

*第九章*，*多 CPU 训练*，提供了如何在单台机器上使用通用方法和 Intel oneCCL 来编写和执行多 CPU 分布式训练的概述。

*第十章*，*使用多个 GPU 进行训练*，提供了如何在单台机器的多 GPU 环境中编码和执行分布式训练的概述。本章介绍了多 GPU 环境的主要特征，并解释了如何使用 NCCL 在多个 GPU 上编码和启动分布式训练，NCCL 是 NVIDIA GPU 的默认通信后端。

*第十一章*，*使用多台机器进行训练*，提供了如何在多个 GPU 和多台机器上进行分布式训练的概述。除了对计算集群的简介解释外，本章还展示了如何使用 Open MPI 作为启动器和 NCCL 作为通信后端，在多台机器之间编码和启动分布式训练。

# 要充分利用本书

您需要了解机器学习、PyTorch 和 Python 的基础知识。

| **书中涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| PyTorch 2.X | Windows、Linux 或 macOS |

如果您使用本书的数字版本，建议您自己键入代码或者从本书的 GitHub 存储库中获取代码（下一节提供链接）。这样做将有助于避免与复制粘贴代码相关的任何潜在错误。

# 下载示例代码文件

您可以从 GitHub 下载本书的示例代码文件，网址为[`github.com/PacktPublishing/Accelerate-Model-Training-with-PyTorch-2.X`](https://github.com/PacktPublishing/Accelerate-Model-Training-with-PyTorch-2.X)。如果代码有更新，将在 GitHub 存储库中更新。

我们还提供其他来自我们丰富书籍和视频目录的代码包，请查阅[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。

# 使用的约定

在本书中使用了许多文本约定。

`文本中的代码`：表示文本中的代码字词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。以下是一个例子：“`ipex.optimize`函数返回模型的优化版本。”

代码块设置如下：

```py
config_list = [{    'op_types': ['Linear'],
    'exclude_op_names': ['layer4'],
    'sparse_ratio': 0.3
}]
```

当我们希望引起您对代码块特定部分的注意时，相关行或项目以粗体显示：

```py
def forward(self, x):    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc2(out)
    return out
```

任何命令行输入或输出如下所示：

```py
maicon@packt:~$ nvidia-smi topo -p -i 0,1Device 0 is connected to device 1 by way of multiple PCIe
```

**粗体**：表示一个新术语、一个重要单词或者在屏幕上显示的单词。例如，菜单或对话框中的单词会以**粗体**显示。以下是一个例子：“**OpenMP**是一个库，用于通过使用多线程技术利用多核处理器的全部性能来并行化任务。”

提示或重要注释

像这样显示。

# 联系我们

我们随时欢迎读者的反馈。

**总体反馈**：如果您对本书的任何方面有疑问，请发送电子邮件至 customercare@packtpub.com，并在邮件主题中提及书名。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误不可避免。如果您在本书中发现错误，我们将不胜感激您向我们报告。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)并填写表格。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法拷贝，请向我们提供位置地址或网站名称。请通过 copyright@packt.com 与我们联系，并提供链接至该材料的链接。

**如果您有兴趣成为作者**：如果您在某个专题上有专业知识，并且有意撰写或为一本书作贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

一旦您阅读了*Accelerate Model Training with PyTorch 2.X*，我们很想听听您的想法！请[点击此处直接访问亚马逊评论页面](https://packt.link/r/1-805-12010-7)并分享您的反馈。

您的评论对我们和技术社区都很重要，将帮助我们确保我们提供的内容质量优秀。

# 下载本书的免费 PDF 副本

感谢您购买本书！

您喜欢随时随地阅读，但无法随身携带印刷书籍吗？

您的电子书购买是否与您选择的设备兼容？

别担心，现在每本 Packt 图书您都可以免费获取一个无 DRM 的 PDF 版本。

随时随地、任何地点、任何设备阅读。直接从您喜爱的技术书籍中搜索、复制和粘贴代码到您的应用程序中。

这些好处并不止于此，您还可以独家获取折扣、新闻通讯和每天收到的优质免费内容

遵循以下简单步骤获取这些好处：

1.  扫描下方的二维码或访问以下链接

![](img/B20959_QR_Free_PDF.jpg)

[`packt.link/free-ebook/978-1-80512-010-0`](https://packt.link/free-ebook/978-1-80512-010-0)

1.  提交您的购书证明

1.  就是这样！我们将免费的 PDF 文件和其他好处直接发送到您的电子邮件中