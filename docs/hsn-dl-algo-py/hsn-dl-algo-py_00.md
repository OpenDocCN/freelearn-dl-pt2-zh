# 前言

深度学习是人工智能领域最受欢迎的领域之一，允许你开发复杂程度各异的多层模型。本书介绍了从基础到高级的流行深度学习算法，并展示了如何使用 TensorFlow 从头开始实现它们。在整本书中，你将深入了解每个算法背后的数学原理及其最佳实现方式。

本书首先解释如何构建自己的神经网络，然后介绍了强大的 Python 机器学习和深度学习库 TensorFlow。接下来，你将快速掌握诸如 NAG、AMSGrad、AdaDelta、Adam、Nadam 等梯度下降变体的工作原理。本书还将为你揭示**循环神经网络**（**RNNs**）和**长短期记忆网络**（**LSTM**）的工作方式，并教你如何使用 RNN 生成歌词。接着，你将掌握卷积网络和胶囊网络的数学基础，这些网络广泛用于图像识别任务。最后几章将带你了解机器如何通过 CBOW、skip-gram 和 PV-DM 理解单词和文档的语义，以及探索各种 GAN（如 InfoGAN 和 LSGAN）和自编码器（如收缩自编码器、VAE 等）的应用。

本书结束时，你将具备在自己的项目中实现深度学习所需的技能。

# 本书适合谁

如果你是机器学习工程师、数据科学家、AI 开发者或者任何希望专注于神经网络和深度学习的人，这本书适合你。完全不熟悉深度学习，但具有一定机器学习和 Python 编程经验的人也会发现这本书很有帮助。

# 本书涵盖的内容

第一章，*深度学习介绍*，解释了深度学习的基础知识，帮助我们理解人工神经网络及其学习过程。我们还将学习如何从头开始构建我们的第一个人工神经网络。

第二章，*TensorFlow 初探*，帮助我们了解最强大和流行的深度学习库之一——TensorFlow。你将了解 TensorFlow 的几个重要功能，并学习如何使用 TensorFlow 构建神经网络以执行手写数字分类。

第三章，*梯度下降及其变种*，深入理解了梯度下降算法。我们将探索几种梯度下降算法的变种，如随机梯度下降（SGD）、Adagrad、ADAM、Adadelta、Nadam 等，并学习如何从头开始实现它们。

第四章，*使用 RNN 生成歌词*，描述了如何使用 RNN 建模顺序数据集以及它如何记住先前的输入。我们将首先对 RNN 有一个基本的理解，然后深入探讨其数学。接下来，我们将学习如何在 TensorFlow 中实现 RNN 来生成歌词。

第五章，*改进 RNN*，开始探索 LSTM 以及它如何克服 RNN 的缺点。稍后，我们将了解 GRU 单元以及双向 RNN 和深层 RNN 的工作原理。在本章末尾，我们将学习如何使用 seq2seq 模型进行语言翻译。

第六章，*揭秘卷积网络*，帮助我们掌握卷积神经网络的工作原理。我们将探索 CNN 前向和反向传播的数学工作方式。我们还将学习各种 CNN 和胶囊网络的架构，并在 TensorFlow 中实现它们。

第七章，*学习文本表示*，涵盖了称为 word2vec 的最新文本表示学习算法。我们将探索 CBOW 和 skip-gram 等不同类型的 word2vec 模型的数学工作方式。我们还将学习如何使用 TensorBoard 可视化单词嵌入。稍后，我们将了解用于学习句子表示的 doc2vec、skip-thoughts 和 quick-thoughts 模型。

第八章，*使用 GAN 生成图像*，帮助我们理解最流行的生成算法之一 GAN。我们将学习如何在 TensorFlow 中实现 GAN 来生成图像。我们还将探索不同类型的 GAN，如 LSGAN 和 WGAN。

第九章，*深入了解 GAN*，揭示了各种有趣的不同类型的 GAN。首先，我们将学习 CGAN，它条件生成器和鉴别器。然后我们看到如何在 TensorFlow 中实现 InfoGAN。接下来，我们将学习如何使用 CycleGAN 将照片转换为绘画作品，以及如何使用 StackGAN 将文本描述转换为照片。

第十章，*使用自编码器重构输入*，描述了自编码器如何学习重构输入。我们将探索并学习如何在 TensorFlow 中实现不同类型的自编码器，如卷积自编码器、稀疏自编码器、收缩自编码器、变分自编码器等。

第十一章，*探索少样本学习算法*，描述如何构建模型从少量数据点中学习。我们将了解什么是少样本学习，并探索流行的少样本学习算法，如孪生网络、原型网络、关系网络和匹配网络。

# 要从本书中获取最大收益

对于那些完全新手于深度学习，但在机器学习和 Python 编程方面有些经验的人，本书将会很有帮助。

# 下载示例代码文件

您可以从您的帐户在[www.packt.com](http://www.packt.com)下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，将文件直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择支持选项卡。

1.  单击代码下载和勘误。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的以下软件解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上：[`github.com/PacktPublishing/Hands-On-Deep-Learning-Algorithms-with-Python`](https://github.com/PacktPublishing/Hands-On-Deep-Learning-Algorithms-with-Python)。如果代码有更新，将在现有的 GitHub 仓库上更新。

我们还提供来自丰富图书和视频目录的其他代码包，可以在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上查看！

# 下载彩色图像

我们还提供一个 PDF 文件，其中包含本书中使用的截屏/图表的彩色图像。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789344158_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789344158_ColorImages.pdf)。

# 使用的约定

本书使用了许多文本约定。

`CodeInText`：指示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。例如："计算`J_plus`和`J_minus`。"

代码块设置如下：

```py
J_plus = forward_prop(x, weights_plus) 
J_minus = forward_prop(x, weights_minus) 
```

任何命令行的输入或输出如下所示：

```py
tensorboard --logdir=graphs --port=8000
```

**粗体**：指示一个新术语、重要词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如此。例如："输入层和输出层之间的任何一层称为**隐藏层**。"

警告或重要说明如下。

提示和技巧以这种方式出现。

# 联系我们

我们非常欢迎读者的反馈。

**常规反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提到书名，并发送邮件至`customercare@packtpub.com`。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误还是可能发生。如果您在本书中发现了错误，我们将不胜感激您向我们报告。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击“勘误提交表格”链接，并填写详细信息。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，请提供给我们位置地址或网站名称。请通过 `copyright@packt.com` 发送包含该材料链接的邮件联系我们。

**如果您有兴趣成为作者**：如果您在某个专题上拥有专业知识，并且有意撰写或为书籍贡献内容，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评论。在您阅读并使用了本书之后，为什么不在您购买它的网站上留下一条评论呢？潜在的读者可以通过您的客观意见做出购买决策，我们在 Packt 可以了解您对我们产品的看法，而我们的作者也可以看到您对他们书籍的反馈。谢谢！

欲了解更多有关 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
