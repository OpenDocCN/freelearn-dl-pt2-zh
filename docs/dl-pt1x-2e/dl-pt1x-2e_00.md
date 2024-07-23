# 前言

PyTorch 因其易用性、高效性以及更符合 Python 开发方式而吸引了深度学习研究人员和数据科学专业人员的关注。本书将帮助您快速掌握 PyTorch 这一最尖端的深度学习库。

在第二版中，您将了解使用 PyTorch 1.x 库的新功能和提供的各种基础构建模块，以推动现代深度学习的发展。您将学习如何使用卷积神经网络（**CNNs**）、循环神经网络（**RNNs**）和长短期记忆网络（**LSTM**）解决实际问题。接着，您将掌握各种最先进的现代深度学习架构的概念，如 ResNet、DenseNet 和 Inception。您将学习如何将神经网络应用于计算机视觉、自然语言处理（**NLP**）等各个领域。您将了解如何使用 PyTorch 构建、训练和扩展模型，并深入探讨生成网络和自编码器等复杂神经网络。此外，您还将了解 GPU 计算以及如何利用 GPU 进行大规模计算。最后，您将学习如何使用基于深度学习的架构解决迁移学习和强化学习问题。

在本书的最后，您将能够轻松在 PyTorch 中实现深度学习应用。

# 本书适合谁

本书适合希望使用 PyTorch 1.x 探索深度学习算法的数据科学家和机器学习工程师。那些希望迁移到 PyTorch 1.x 的人会发现本书富有洞见。为了充分利用本书，具备 Python 编程的工作知识和一些机器学习知识将非常有帮助。

# 本书内容涵盖了什么

第一章，*使用 PyTorch 开始深度学习*，介绍了深度学习、机器学习和人工智能的历史。本章涵盖了它们与神经科学以及统计学、信息理论、概率论和线性代数等科学领域的关系。

第二章，*神经网络的构建模块*，涵盖了使用 PyTorch 理解和欣赏神经网络所需的各种数学概念。

第三章，*深入探讨神经网络*，向您展示如何将神经网络应用于各种现实场景。

第四章，*计算机视觉中的深度学习*，涵盖了现代 CNN 架构的各种构建模块。

第五章，*使用序列数据进行自然语言处理*，向您展示如何处理序列数据，特别是文本数据，并教您如何创建网络模型。

第六章，*实现自编码器*，通过自编码器的介绍介绍了半监督学习算法的概念。还涵盖了如何使用受限玻尔兹曼机理解数据的概率分布。

第七章，*生成对抗网络的应用*，展示了如何构建能够生成文本和图像的生成模型。

第八章，*现代网络架构下的迁移学习*，介绍了现代架构如 ResNet、Inception、DenseNet 和 Seq2Seq，并展示了如何使用预训练权重进行迁移学习。

第九章，*深度强化学习*，从强化学习的基本介绍开始，包括代理、状态、动作、奖励和策略的覆盖。还包括基于深度学习的强化学习问题的实用代码，如 Deep Q 网络、策略梯度方法和演员-评论家模型。

第十章，*接下来做什么？*，快速概述了本书涵盖的内容，并提供了如何跟上领域最新进展的信息。

# 要充分利用这本书

熟悉 Python 将会很有帮助。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](https://www.packtpub.com/support)，并注册以直接通过电子邮件获取文件。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packt.com](http://www.packt.com)。

1.  选择“支持”选项卡。

1.  点击“代码下载”。

1.  在搜索框中输入书名并按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压软件解压缩文件夹：

+   Windows 使用 WinRAR/7-Zip

+   Mac 使用 Zipeg/iZip/UnRarX

+   Linux 使用 7-Zip/PeaZip

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Deep-Learning-with-PyTorch-1.x`](https://github.com/PacktPublishing/Deep-Learning-with-PyTorch-1.x)。如果代码有更新，将在现有的 GitHub 仓库中更新。

我们还提供了来自我们丰富图书和视频目录的其他代码包，都可以在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**查看！

# 下载彩色图像

我们还提供了一份包含本书中使用的屏幕截图/图表的彩色图像的 PDF 文件。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781838553005_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781838553005_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：指示文本中的代码词汇，数据库表名，文件夹名称，文件名，文件扩展名，路径名，虚拟 URL，用户输入和 Twitter 句柄。例如：“让我们使用简单的 Python 函数，如 `split` 和 `list`，将文本转换为标记。”

代码块设置如下：

```py
toy_story_review = "Just perfect. Script, character, animation....this manages to break free of the yoke of 'children's movie' to simply be one of the best movies of the 90's, full-stop."

print(list(toy_story_review))
```

当我们希望引起您对代码块特定部分的注意时，相关行或项将加粗显示：

```py
['J', 'u', 's', 't', ' ', 'p', 'e', 'r', 'f', 'e', 'c', 't', '.', ' ', 'S', 'c', 'r', 'i', 'p', 't', ',', ' ', 'c', 'h', 'a', 'r', 'a', 'c', 't', 'e', 'r', ',', ' ', 'a', 'n', 'i', 'm', 'a', 't', 'i', 'o', 'n', '.', '.', '.', '.', 't', 'h', 'i', 's', ' ', 'm', 'a', 'n', 'a', 'g', 'e', 's', ' ', 't', 'o', ' ', 'b', 'r', 'e', 'a', 'k', ' ', 'f', 'r', 'e', 'e', ' ', 'o', 'f', ' ', 't', 'h', 'e', ' ', 'y', 'o', 'k', 'e', ' ', 'o', 'f', ' ', "'", 'c', 'h', 'i', 'l', 'd', 'r', 'e', 'n', "'", 's', ' ', 'm', 'o', 'v', 'i', 'e', "'", ' ', 't', 'o', ' ', 's', 'i', 'm', 'p', 'l', 'y', ' ', 'b', 'e', ' ', 'o', 'n', 'e', ' ', 'o', 'f', ' ', 't', 'h', 'e', ' ', 'b', 'e', 's', 't', ' ', 'm', 'o', 'v', 'i', 'e', 's', ' ', 'o', 'f', ' ', 't', 'h', 'e', ' ', '9', '0', "'", 's', ',', ' ', 'f', 'u', 'l', 'l', '-', 's', 't', 'o', 'p', '.']
```

任何命令行输入或输出都写成以下格式：

```py
pip install torchtext
```

**粗体**：表示新术语，重要词汇或屏幕上显示的词语。例如，菜单或对话框中的词语在文本中显示为这样。这是一个例子：“我们将帮助您理解**递归神经网络**（**RNNs**）。”

警告或重要提示看起来像这样。

小贴士和技巧看起来像这样。

# 联系我们

我们非常欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在消息主题中提及书名，并发送电子邮件至 `customercare@packtpub.com`。

**勘误**：尽管我们已尽一切努力确保内容的准确性，但错误偶尔也会发生。如果您在本书中发现错误，请向我们报告。请访问 [www.packtpub.com/support/errata](https://www.packtpub.com/support/errata)，选择您的书籍，点击勘误提交表单链接，并填写详细信息。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法副本，我们将不胜感激，如果您能提供给我们具体位置或网站名称的信息。请联系我们，发送至 `copyright@packt.com`，并附上材料的链接。

**如果您有兴趣成为作者**：如果您对某个您专业的主题感兴趣，并且您有意参与撰写或贡献书籍，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评论。一旦您阅读并使用了本书，请为您购买的网站留下评论，以便潜在读者可以看到并使用您的客观意见来做出购买决策，我们在 Packt 可以了解您对我们产品的看法，而我们的作者可以看到您对他们书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问 [packt.com](http://www.packt.com/)。