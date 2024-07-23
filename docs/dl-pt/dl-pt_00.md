# 前言

PyTorch 因其灵活性和易用性而引起了数据科学专业人士和深度学习从业者的关注。本书介绍了深度学习和 PyTorch 的基本构建模块，并展示了如何使用实用的方法解决实际问题。你还将学习到一些现代架构和技术，用于解决一些前沿研究问题。

本书提供了各种最先进的深度学习架构（如 ResNet、DenseNet、Inception 和 Seq2Seq）的直觉，而不深入数学。它还展示了如何进行迁移学习，如何利用预计算特征加快迁移学习，以及如何使用嵌入、预训练嵌入、LSTM 和一维卷积进行文本分类。

通过本书，你将成为一名熟练的深度学习从业者，能够使用所学的不同技术解决一些商业问题。

# 本书的受众

本书适合工程师、数据分析师和数据科学家，对深度学习感兴趣，以及那些希望探索和实施 PyTorch 高级算法的人群。了解机器学习有所帮助，但不是必须的。预期具备 Python 编程知识。

# 本书内容涵盖

第一章，*使用 PyTorch 开始深度学习*，回顾了**人工智能**（**AI**）和机器学习的历史，并观察了深度学习的最近发展。我们还将探讨硬件和算法的各种改进如何在不同应用中实现深度学习的巨大成功。最后，我们将介绍由 Facebook 基于 Torch 开发的优秀 PyTorch Python 库。

第二章，*神经网络的基本构建模块*，讨论了 PyTorch 的各种构建模块，如变量、张量和`nn.module`，以及它们如何用于开发神经网络。

第三章，*深入理解神经网络*，涵盖了训练神经网络的不同过程，如数据准备、用于批处理张量的数据加载器、`torch.nn`包创建网络架构以及使用 PyTorch 损失函数和优化器。

第四章，*机器学习基础*，涵盖了不同类型的机器学习问题，以及诸如过拟合和欠拟合等挑战。我们还介绍了数据增强、添加 Dropout 以及使用批归一化等不同技术来防止过拟合。

第五章，*计算机视觉中的深度学习*，讲解了**卷积神经网络**（**CNNs**）的构建模块，如一维和二维卷积、最大池化、平均池化、基本 CNN 架构、迁移学习以及使用预先卷积特征进行更快训练。

第六章，*序列数据和文本的深度学习*，涵盖了词嵌入、如何使用预训练的嵌入、RNN、LSTM 和一维卷积进行 IMDB 数据集的文本分类。

第七章，*生成网络*，讲解如何利用深度学习生成艺术图像，使用 DCGAN 生成新图像，以及使用语言建模生成文本。

第八章，*现代网络架构*，探讨了现代计算机视觉应用的架构，如 ResNet、Inception 和 DenseNet。我们还将快速介绍编码器-解码器架构，该架构驱动了现代系统，如语言翻译和图像字幕。

第九章，*接下来做什么？*，总结了我们所学的内容，并探讨如何在深度学习领域保持更新。

# 要充分利用本书

本书所有章节（除了第一章，*使用 PyTorch 入门深度学习*和第九章，*接下来做什么？*）在书籍的 GitHub 仓库中都有关联的 Jupyter Notebooks。为了节省空间，文本中可能未包含代码运行所需的导入。您应该能够从 Notebooks 中运行所有代码。

本书侧重于实际示例，因此在阅读章节时运行 Jupyter Notebooks。

使用带有 GPU 的计算机将有助于快速运行代码。有一些公司，如 [paperspace.com](https://www.paperspace.com/) 和 [www.crestle.com](https://www.crestle.com/)，简化了运行深度学习算法所需的复杂性。

# 下载示例代码文件

您可以从 [www.packtpub.com](http://www.packtpub.com) 的帐户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问 [www.packtpub.com/support](http://www.packtpub.com/support) 并注册，以便直接通过电子邮件获取文件。

您可以按照以下步骤下载代码文件：

1.  在 [www.packtpub.com](http://www.packtpub.com/support) 登录或注册。

1.  选择“支持”选项卡。

1.  单击“代码下载与勘误”。

1.  在搜索框中输入书名并按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上：[`github.com/PacktPublishing/Deep-Learning-with-PyTorch`](https://github.com/PacktPublishing/Deep-Learning-with-PyTorch)。如果代码有更新，将在现有的 GitHub 存储库中更新。

我们还有其他来自我们丰富书目和视频的代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**查看！请查看！

# 下载彩色图像

我们还提供了一个包含本书使用的屏幕截图/图表的彩色图像的 PDF 文件。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/DeepLearningwithPyTorch_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/DeepLearningwithPyTorch_ColorImages.pdf)

# 用法约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码字词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。以下是一个示例："自定义类必须实现两个主要函数，即`__len__(self)`和`__getitem__(self, idx)`。"

代码块设置如下：

```py
x,y = get_data() # x - represents training data,y -                 represents target variables

w,b = get_weights() # w,b - Learnable parameters

for i in range(500):
    y_pred = simple_network(x) # function which computes wx + b
    loss = loss_fn(y,y_pred) # calculates sum of the squared differences of y and y_pred

if i % 50 == 0: 
        print(loss)
    optimize(learning_rate) # Adjust w,b to minimize the loss
```

任何命令行输入或输出都是这样写的：

```py
conda install pytorch torchvision cuda80 -c soumith
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。

警告或重要说明看起来像这样。

提示和技巧看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**总体反馈**：请发送电子邮件至`feedback@packtpub.com`，在主题中提到书名。如果您对本书的任何方面有疑问，请发送电子邮件至`questions@packtpub.com`。

**勘误**：尽管我们已尽一切努力确保内容的准确性，但错误难免。如果您在本书中发现错误，请告知我们。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法副本，请向我们提供位置地址或网站名称。请联系我们，地址为`copyright@packtpub.com`，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个专题上有专业知识，并且有兴趣撰写或贡献一本书，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。阅读并使用本书后，请为什么不在您购买它的网站上留下评论呢？潜在的读者可以看到并使用您的公正意见来做购买决策，我们在 Packt 可以了解您对我们产品的看法，我们的作者可以看到您对他们书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问[packtpub.com](https://www.packtpub.com/)。
