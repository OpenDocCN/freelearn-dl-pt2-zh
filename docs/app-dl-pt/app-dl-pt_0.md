# 前言

## 关于

本节简要介绍了作者、本书的内容覆盖范围、开始所需的技术技能以及完成所有包含的活动和练习所需的硬件和软件要求。

## 关于本书

机器学习正迅速成为解决数据问题的首选方式，这要归功于大量的数学算法，这些算法可以发现我们看不见的模式。

*应用深度学习与 PyTorch* 将带领您深入了解深度学习及其算法和应用。本书从帮助您浏览深度学习和 PyTorch 的基础开始。一旦您熟悉了 PyTorch 语法并能够构建单层神经网络，您将逐步学习通过配置和训练卷积神经网络（CNN）来解决更复杂的数据问题。随着章节的推进，您将发现如何通过实现递归神经网络（RNN）来解决自然语言处理问题。

在本书结束时，您将能够应用您在学习过程中积累的技能和信心，使用 PyTorch 构建深度学习解决方案，解决您的业务数据问题。

### 关于作者

**海雅特·萨莱** 毕业于商业管理专业后，发现数据分析对理解和解决现实生活问题的重要性。此后，作为一名自学者，她不仅为全球多家公司担任机器学习自由职业者，还创立了一家旨在优化日常流程的人工智能公司。她还撰写了由 Packt Publishing 出版的《机器学习基础》。

### 目标

+   检测多种数据问题，可以应用深度学习解决方案

+   学习 PyTorch 语法并用其构建单层神经网络

+   构建一个深度神经网络以解决分类问题

+   开发风格迁移模型

+   实施数据增强并重新训练您的模型

+   使用递归神经网络构建文本处理系统

### 受众

*应用深度学习与 PyTorch* 适用于希望使用深度学习技术处理数据的数据科学家、数据分析师和开发人员。任何希望探索并实施 PyTorch 高级算法的人都会发现本书有用。具备 Python 的基本知识和机器学习基础是必需的。然而，了解 NumPy 和 pandas 将是有益但不是必要的。

### 方法

*应用深度学习与 PyTorch* 采用实际操作的方式，每章节都有一个完整的实例，从数据获取到结果解释全过程演示。考虑到所涉及概念的复杂性，各章节包含多个图形表示以促进学习。

### 硬件要求

为了最佳学习体验，我们建议使用以下硬件配置：

+   处理器：Intel Core i3 或同等级别

+   内存：4 GB RAM

+   存储空间：35 GB 可用空间

### 软件需求

您还需要提前安装以下软件：

+   操作系统：Windows 7 SP1 64 位、Windows 8.1 64 位或 Windows 10 64 位、Ubuntu Linux 或 OS X 的最新版本

+   浏览器：Google Chrome/Mozilla Firefox 最新版本

+   Notepad++/Sublime Text 作为 IDE（可选，因为您可以使用浏览器中的 Jupyter 笔记本练习所有内容）

+   已安装 Python 3.4+（最新版本为 Python 3.7）（来自 [`python.org`](https://python.org)）

+   需要的 Python 库（Jupyter、Numpy、Pandas、Matplotlib、BeautifulSoup4 等）

### 约定

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下："我们在这里使用 `requires_grad` 参数告诉 PyTorch 计算该张量的梯度。"

代码块如下所示：

```py
a = torch.tensor([5.0, 3.0], requires_grad=True)
b = torch.tensor([1.0, 4.0])
ab = ((a + b) ** 2).sum()
ab.backward()
```

新术语和重要单词显示为粗体。屏幕上看到的单词，例如菜单或对话框中的内容，以如下形式出现在文本中："要下载将使用的数据集，请访问 [`archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients`](http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)，然后点击 `.xls` 文件。"

### 安装代码包

将课程的代码包复制到您本地计算机上的一个文件夹，以便在学习本书时轻松访问。确切的位置取决于您操作系统的限制和个人偏好。

在本书的 GitHub 仓库中（[`github.com/TrainingByPackt/Applied-Deep-Learning-with-PyTorch`](https://github.com/TrainingByPackt/Applied-Deep-Learning-with-PyTorch)），您可以找到一个 `requirements.txt` 文件，其中包含本书不同活动和练习所需的所有库和模块列表及其版本。

### 其他资源

本书的代码包也托管在 GitHub 上，链接为 [`github.com/TrainingByPackt/Applied-Deep-Learning-with-PyTorch`](https://github.com/TrainingByPackt/Applied-Deep-Learning-with-PyTorch)。

我们还有其他代码包，来自我们丰富的书籍和视频目录，可在 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/) 查看！
