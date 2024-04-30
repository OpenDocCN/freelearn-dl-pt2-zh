# 序言

随着全球对人工智能的兴趣不断增长，深度学习引起了广泛的关注。每天，深度学习算法被广泛应用于不同行业。本书将为您提供关于主题的所有实际信息，包括最佳实践，使用真实用例。您将学会识别和提取信息，以提高预测精度并优化结果。

书籍首先快速回顾了重要的机器学习概念，然后直接深入探讨使用 scikit-learn 的深度学习原则。随后，您将学习使用最新的开源库，如 Theano、Keras、Google 的 TensorFlow 和 H2O。使用本指南来揭示模式识别的困难之处，以更高精度扩展数据，并讨论深度学习算法和技术。无论您是想深入了解深度学习，还是想探索如何更充分地利用这一强大技术，您都可以在本书中找到答案。

# 本书内容

第一章, *机器学习 - 简介*，介绍了不同的机器学习方法和技术，以及它们在现实问题中的一些应用。我们将介绍 Python 中用于机器学习的一个主要开源软件包，即 scikit-learn。

第二章, *神经网络*，正式介绍了神经网络是什么。我们将深入描述神经元的工作原理，并展示如何堆叠多层来创建和使用深度前馈神经网络。

第三章, *深度学习基础*，将带您了解深度学习是什么，以及它与深度神经网络的关系。

第四章, *无监督特征学习*，涵盖了两种最强大且常用的无监督特征学习架构：自编码器和受限玻尔兹曼机。

第五章, *图像识别*，从类比我们视觉皮层的工作方式开始，并介绍卷积层，随后描述了它们为什么有效的直观认识。

第六章, *循环神经网络和语言模型*，讨论了一些非常有前景的强大方法，在许多任务中表现出很高的潜力，比如语言建模和语音识别。

第七章, *棋盘游戏的深度学习*，介绍了用于解决跳棋和国际象棋等棋盘游戏的不同工具。

第八章, *计算机游戏的深度学习*，研究了训练 AI 玩计算机游戏所面临的更复杂的问题。

第九章, *异常检测*，从解释异常值检测和异常检测概念之间的差异和相似之处开始。您将通过一个想象中的欺诈案例研究，以及展示在现实世界应用程序中存在异常的危险以及自动化和快速检测系统的重要性的示例来指导您。

第十章, *构建一个生产就绪的入侵检测系统*，利用 H2O 和常用做法构建一个可部署到生产环境中的可扩展分布式系统。您将学习如何使用 Spark 和 MapReduce 来训练深度学习网络，如何使用自适应学习技术实现更快的收敛速度，并且非常重要的是如何验证模型和评估端到端流程。

# 您需要为本书准备什么

您将能够使用以下任何操作系统：Windows，Linux 和 Macintosh。

要顺利阅读本书，您需要以下内容：

+   TensorFlow

+   Theano

+   Keras

+   Matplotlib

+   H2O 。

+   scikit-learn

# 这本书是为谁准备的

本书适用于数据科学实践者和拥有基本机器学习概念和一些 Python 编程经验的有志者。还希望具备数学背景，对微积分和统计概念有概念上的理解。

# 约定

在本书中，您将找到一系列区分不同类型信息的文本样式。以下是一些示例和解释它们的含义。

文本中的代码字、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：上述用于绘图的代码应立即清晰，我们只需要注意导入`cm`这一行。

代码块设置如下：

```py
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train = X_train.reshape(50000, 3072)
X_test = X_test.reshape(10000, 3072)
input_size = 3072
```

当我们想要引起您对代码块的特定部分的注意时，相关行或项将以粗体设置：

```py
def monte_carlo_tree_search_uct(board_state, side, number_of_rollouts):
    state_results = collections.defaultdict(float)
    state_samples = collections.defaultdict(float)
```

任何命令行输入或输出都是这样写的:

```py
git clone https://github.com/fchollet/keras.git
cd keras
python setup.py install

```

**新术语**和**重要单词**以粗体显示。

### 注意

警告或重要说明以这样的框显示。

### 提示

提示和技巧显示如下。

# 读者反馈

我们一直欢迎读者的反馈意见。请告诉我们你对这本书的看法，你喜欢或不喜欢什么。读者的反馈对我们很重要，因为它可以帮助我们开发真正让你受益的书籍。

要发送一般性反馈意见，只需通过电子邮件发送到`<feedback@packtpub.com>`，并在主题中提到书的标题。

如果您在某个专业领域有专长，并且有兴趣写作或为书籍作出贡献，请参阅我们的作者指南网址[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在您已经是 Packt 书籍的自豪所有者，我们有一些事情可以帮助您从中获益最大化。

## 下载示例代码

您可以从[`www.packtpub.com`](http://www.packtpub.com)的您的账户中下载该书的示例代码文件。如果您在其他地方购买了本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)注册并直接向您发送文件。

您可以按照以下步骤下载代码文件：

+   使用您的电子邮件地址和密码登录或注册我们的网站。

+   将鼠标指针悬停在顶部的**SUPPORT**选项卡上。

+   单击**Code Downloads & Errata**。

+   在搜索框中输入书名。

+   选择您想要下载代码文件的书籍。

+   从下拉菜单中选择您购买本书的地点。

+   单击**Code Download**。

你也可以通过单击 Packt Publishing 网站上这本书的网页上的**Code Files**按钮来下载代码文件。你可以通过在**Search**框中输入书名来访问这个页面。请注意，你需要登录你的 Packt 账户。

下载文件后，请确保使用最新版本的解压软件对文件进行解压或提取文件夹。

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

该书的代码完整包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Python-Deep-Learning`](https://github.com/PacktPublishing/Python-Deep-Learning)。我们还提供来自我们丰富图书和视频目录的其他代码完整包，请查看。

## 下载本书的彩色插图

我们还为您提供了一个 PDF 文件，其中包含本书使用的屏幕截图/图表的彩色图片。彩色图片将帮助您更好地理解输出中的变化。您可以从[`www.packtpub.com/sites/default/files/downloads/PythonDeepLearning_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/PythonDeepLearning_ColorImages.pdf)下载此文件。

# 勘误

尽管我们已经竭尽全力确保内容的准确性，但是错误是难以避免的。如果您在我们的书籍中发现错误，也许是文本或代码错误，我们将非常感谢您能向我们报告。通过这样做，您可以帮助其他读者避免困惑，并帮助我们改进后续版本的这本书。如果您发现任何勘误，请访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表格**链接，然后输入您的勘误信息。一旦您的勘误被验证，您的提交将被接受，勘误将被上传到我们的网站，或在标题下的勘误部分中添加到任何现有勘误列表中。

要查看先前提交的勘误表，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索框中输入书籍名称。所需信息将显示在**勘误表**部分下。

## 盗版

互联网上盗版版权材料是一个跨媒体持续存在的问题。在 Packt，我们非常重视对我们的版权和许可的保护。如果您在互联网上发现我们作品的任何非法副本，请立即向我们提供位置地址或网站名称，以便我们采取措施。

请通过链接<copyright@packtpub.com>与我们联系，提供涉嫌盗版材料的链接。

我们感谢您在保护我们的作者和为您带来有价值的内容方面所做的帮助。

## 问题

如果您对这本书的任何方面有问题，可以通过邮件联系我们\<questions@packtpub.com\>，我们将尽力解决问题。
