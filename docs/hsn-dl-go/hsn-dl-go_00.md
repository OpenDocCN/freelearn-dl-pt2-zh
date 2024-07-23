# 序言

Go 是由 Google 设计的开源编程语言，旨在高效处理大型项目。它使得构建可靠、简单和高效的软件变得简单直接。

本书立即进入了在 Go 语言中实现**深度神经网络**（**DNNs**）的实用性方面。简单来说，书名已包含其目的。这意味着书中将涉及大量的技术细节、大量代码以及（不算太多的）数学。当你最终合上书本或关闭 Kindle 时，你将知道如何（以及为什么）实现现代可扩展的 DNNs，并能够根据自己在任何行业或疯狂科学项目中的需求重新利用它们。

# 本书适合谁

本书适合数据科学家、机器学习工程师和深度学习爱好者，他们希望将深度学习引入其 Go 应用程序中。预计读者熟悉机器学习和基本的 Golang 代码，以便从本书中获益最大化。

# 本书内容涵盖了什么

第一章，*在 Go 中深度学习简介*，介绍了深度学习的历史和应用。本章还概述了使用 Go 进行机器学习的情况。

第二章，*什么是神经网络及如何训练？*，介绍了如何构建简单的神经网络，以及如何检查图形，还涵盖了许多常用的激活函数。本章还讨论了用于神经网络的梯度下降算法的不同选项和优化。

第三章，*超越基础神经网络 – 自编码器和 RBM*，展示了如何构建简单的多层神经网络和一个自编码器。本章还探讨了一个概率图模型，即用于无监督学习创建电影推荐引擎的 RBM 的设计和实现。

第四章，*CUDA – GPU 加速训练*，探讨了深度学习的硬件方面，以及 CPU 和 GPU 如何满足我们的计算需求。

第五章，*基于递归神经网络的下一个词预测*，深入探讨了基本 RNN 的含义及其训练方法。您还将清楚地了解 RNN 架构，包括 GRU/LSTM 网络。

第六章，*卷积神经网络进行对象识别*，向您展示如何构建 CNN 以及如何调整一些超参数（如 epoch 数量和批处理大小）以获得所需的结果，并在不同计算机上顺利运行。

第七章，*使用深度 Q 网络解决迷宫*，介绍了强化学习和 Q-learning，以及如何构建 DQN 来解决迷宫问题。

第八章，*使用变分自编码器生成模型*，展示了如何构建 VAE，并探讨了 VAE 相对于标准自编码器的优势。本章还展示了如何理解在网络上变化潜在空间维度的影响。

第九章，*构建深度学习管道*，讨论了数据管道的定义及为何使用 Pachyderm 来构建或管理它们。

第十章，*扩展部署*，涉及到 Pachyderm 底层的多种技术，包括 Docker 和 Kubernetes，还探讨了如何利用这些工具将堆栈部署到云基础设施。

# 为了更好地使用本书

本书主要使用 Go 语言，Go 的 Gorgonia 包，Go 的 Cu 包，以及 NVIDIA 提供的支持 CUDA 的 CUDA（加驱动程序）和支持 CUDA 的 NVIDIA GPU。此外，还需要 Docker 用于第三部分，*管道、部署及其他*！

# 下载示例代码文件

您可以从您的帐户在 [www.packt.com](http://www.packt.com) 下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问 [www.packt.com/support](http://www.packt.com/support) 并注册，以便将文件直接发送到您的邮箱。

按照以下步骤下载代码文件：

1.  登录或注册 [www.packt.com](http://www.packt.com)。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名并按照屏幕上的说明操作。

下载完成后，请确保使用最新版本的解压软件解压缩文件夹：

+   Windows 下的 WinRAR/7-Zip

+   Mac 下的 Zipeg/iZip/UnRarX

+   Linux 下的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上：[`github.com/PacktPublishing/Hands-On-Deep-Learning-with-Go`](https://github.com/PacktPublishing/Hands-On-Deep-Learning-with-Go)。如果代码有更新，将在现有的 GitHub 仓库中更新。

我们还有其他来自丰富图书和视频目录的代码包，可以在 [`github.com/PacktPublishing/Hands-On-Deep-Learning-with-Go`](https://github.com/PacktPublishing/Hands-On-Deep-Learning-with-Go) 查看！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图示的彩色图像。你可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789340990_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789340990_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：指示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。以下是一个示例："将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为系统中的另一个磁盘。"

代码块设置如下：

```py
type nn struct {
    g *ExprGraph
    w0, w1 *Node

    pred *Node
}
```

当我们希望引起您对代码块的特定部分的注意时，相关的行或项将以粗体显示：

```py
intercept Ctrl+C
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    doneChan := make(chan bool, 1)
```

任何命令行输入或输出均显示如下：

```py
sudo apt install nvidia-390 nvidia-cuda-toolkit libcupti-dev
```

**粗体**：指示一个新术语、一个重要单词或屏幕上显示的单词。例如，菜单或对话框中的单词在文本中显示为这样。以下是一个示例："从管理面板中选择系统信息。"

警告或重要提示显示如此。

提示和技巧显示如此。

# 联系我们

我们始终欢迎读者的反馈。

**总体反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并发送邮件至`customercare@packtpub.com`。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在本书中发现错误，请向我们报告。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表格链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法复制，请向我们提供位置地址或网站名称，我们将不胜感激。请联系我们，链接为`copyright@packt.com`。

**如果您有兴趣成为作者**：如果您在某个专题上有专业知识并且有意撰写或贡献一本书籍，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评论。一旦您阅读并使用了本书，请在购买它的网站上留下评论。潜在的读者可以看到并使用您的客观意见来做出购买决策，我们在 Packt 能够了解您对我们产品的看法，我们的作者可以看到您对他们书籍的反馈。谢谢！

欲了解更多有关 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
