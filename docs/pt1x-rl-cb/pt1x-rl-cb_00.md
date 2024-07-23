# 前言

强化学习兴起的原因在于它通过学习在环境中采取最优行动来最大化累积奖励的概念，从而革新了自动化。

*PyTorch 1.x 强化学习菜谱*向您介绍了重要的强化学习概念，以及在 PyTorch 中实现算法的方法。本书的每一章都将引导您了解不同类型的强化学习方法及其在行业中的应用。通过包含真实世界示例的配方，您将发现在动态规划、蒙特卡洛方法、时间差分与 Q-learning、多臂老虎机、函数逼近、深度 Q 网络和策略梯度等强化学习技术领域提升知识和熟练度是多么有趣而易于跟随的事情。有趣且易于跟随的示例，如 Atari 游戏、21 点、Gridworld 环境、互联网广告、Mountain Car 和 Flappy Bird，将让您在实现目标之前保持兴趣盎然。

通过本书，您将掌握流行的强化学习算法的实现，并学习将强化学习技术应用于解决其他实际问题的最佳实践。

# 本书适合对象

寻求在强化学习中快速解决不同问题的机器学习工程师、数据科学家和人工智能研究人员会发现本书非常有用。需要有机器学习概念的先验知识，而对 PyTorch 的先前经验将是一个优势。

# 本书内容概述

第一章，*使用 PyTorch 入门强化学习*，是本书逐步指南开始的地方，为那些希望开始学习使用 PyTorch 进行强化学习的读者提供了指导。我们将建立工作环境并熟悉使用 Atari 和 CartPole 游戏场景的强化学习环境。本章还将涵盖几种基本的强化学习算法的实现，包括随机搜索、爬山法和策略梯度。最后，读者还将有机会复习 PyTorch 的基础知识，并为即将到来的学习示例和项目做好准备。

第二章，*马尔可夫决策过程与动态规划*，从创建马尔可夫链和马尔可夫决策过程开始，后者是大多数强化学习算法的核心。然后，我们将介绍两种解决马尔可夫决策过程（MDP）的方法，即值迭代和策略迭代。通过实践策略评估，我们将更加熟悉 MDP 和贝尔曼方程。我们还将逐步演示如何解决有趣的硬币翻转赌博问题。最后，我们将学习如何执行动态规划以扩展学习能力。

第三章，*蒙特卡洛方法进行数值估计*，专注于蒙特卡洛方法。我们将从使用蒙特卡洛估算 pi 值开始。接着，我们将学习如何使用蒙特卡洛方法预测状态值和状态-动作值。我们将展示如何训练一个代理程序在 21 点中获胜。此外，我们将通过开发各种算法探索在线策略的第一次访问蒙特卡洛控制和离线蒙特卡洛控制。还将涵盖带有 epsilon-greedy 策略和加权重要性采样的蒙特卡洛控制。

第四章，*时间差分与 Q 学习*，首先建立了 CliffWalking 和 Windy Gridworld 环境场地，这些将在时间差分和 Q 学习中使用。通过我们的逐步指南，读者将探索用于预测的时间差分，并且会通过 Q 学习获得实际控制经验，以及通过 SARSA 实现在线策略控制。我们还将处理一个有趣的项目，出租车问题，并展示如何使用 Q 学习和 SARSA 算法解决它。最后，我们将涵盖 Double Q-learning 算法作为额外的部分。

第五章，*解决多臂赌博问题*，涵盖了多臂赌博算法，这可能是强化学习中最流行的算法之一。我们将从创建多臂赌博问题开始。我们将看到如何使用四种策略解决多臂赌博问题，包括 epsilon-greedy 策略、softmax 探索、上置信度界算法和 Thompson 采样算法。我们还将处理一个十亿美元的问题，在线广告，展示如何使用多臂赌博算法解决它。最后，我们将开发一个更复杂的算法，上下文赌博算法，并用它来优化显示广告。

第六章，*使用函数逼近扩展学习*，专注于函数逼近，并将从设置 Mountain Car 环境场地开始。通过我们的逐步指南，我们将讨论为什么使用函数逼近而不是表查找，并且通过 Q 学习和 SARSA 等现有算法融入函数逼近的实际经验。我们还将涵盖一个高级技术，即使用经验重放进行批处理。最后，我们将介绍如何使用本章学到的内容来解决 CartPole 问题。

第七章，《行动中的深度 Q 网络》，涵盖了深度 Q 学习或**深度 Q 网络**（**DQN**），被认为是最现代的强化学习技术。我们将逐步开发一个 DQN 模型，并了解经验回放和目标网络在实践中使深度 Q 学习发挥作用的重要性。为了帮助读者解决雅达利游戏问题，我们将演示如何将卷积神经网络融入到 DQN 中。我们还将涵盖两种 DQN 变体，分别为双重 DQN 和对战 DQN。我们将介绍如何使用双重 DQN 调优 Q 学习算法。

第八章，《实施策略梯度和策略优化》，专注于策略梯度和优化，并首先实施 REINFORCE 算法。然后，我们将基于 ClifWalking 开发带基准线的 REINFORCE 算法。我们还将实施 actor-critic 算法，并应用它来解决 ClifWalking 问题。为了扩展确定性策略梯度算法，我们从 DQN 中应用技巧，并开发深度确定性策略梯度。作为一个有趣的体验，我们训练一个基于交叉熵方法的代理来玩 CartPole 游戏。最后，我们将谈论如何使用异步 actor-critic 方法和神经网络来扩展策略梯度方法。

第九章，《毕业项目 - 使用 DQN 玩 Flappy Bird》带领我们进行一个毕业项目 - 使用强化学习玩 Flappy Bird。我们将应用本书中学到的知识来构建一个智能机器人。我们将专注于构建一个 DQN，调优模型参数，并部署模型。让我们看看鸟在空中能飞多久。

# 为了从本书中获得最大收益

寻求强化学习中不同问题的快速解决方案的数据科学家、机器学习工程师和人工智能研究人员会发现这本书很有用。需要有机器学习概念的先前接触，而 PyTorch 的先前经验并非必需，但会是一个优势。

# 下载示例代码文件

您可以从您在 [www.packt.com](http://www.packt.com) 的账户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问 [www.packtpub.com/support](https://www.packtpub.com/support) 并注册，以直接通过电子邮件接收文件。

您可以按照以下步骤下载代码文件：

1.  在 [www.packt.com](http://www.packt.com) 登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保您使用最新版本的解压工具解压或提取文件夹：

+   Windows 系统使用 WinRAR/7-Zip

+   Mac 系统使用 Zipeg/iZip/UnRarX

+   Linux 系统使用 7-Zip/PeaZip

The code bundle for the book is also hosted on GitHub at [`github.com/PacktPublishing/PyTorch-1.x-Reinforcement-Learning-Cookbook`](https://github.com/PacktPublishing/PyTorch-1.x-Reinforcement-Learning-Cookbook). In case there's an update to the code, it will be updated on the existing GitHub repository.

We also have other code bundles from our rich catalog of books and videos available at **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**. Check them out!

# Download the color images

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. You can download it here: [`static.packt-cdn.com/downloads/9781838551964_ColorImages.pdf`](https://static.packt-cdn.com/downloads/9781838551964_ColorImages.pdf)[.](http://www.packtpub.com/sites/default/files/downloads/Bookname_ColorImages.pdf)

# Conventions used

There are a number of text conventions used throughout this book.

`CodeInText`：指示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 用户名。例如："所谓的`empty`并不意味着所有元素都有`Null`值。"

A block of code is set as follows:

```py
>>> def random_policy():
...     action = torch.multinomial(torch.ones(n_action), 1).item()
...     return action
```

Any command-line input or output is written as follows:

```py
conda install pytorch torchvision -c pytorch
```

**Bold**: 表示新术语、重要词汇或屏幕上看到的词语。例如，菜单或对话框中的单词以这种方式出现在文本中。例如："这种方法称为**随机搜索**，因为每次试验中权重都是随机选择的，希望通过大量试验找到最佳权重。"

Warnings or important notes appear like this.

Tips and tricks appear like this.

# Sections

In this book, you will find several headings that appear frequently (*Getting ready*, *How to do it...*, *How it works...*, *There's more...*, and *See also*).

To give clear instructions on how to complete a recipe, use these sections as follows:

# Getting ready

This section tells you what to expect in the recipe and describes how to set up any software or any preliminary settings required for the recipe.

# How to do it...

This section contains the steps required to follow the recipe.

# How it works...

This section usually consists of a detailed explanation of what happened in the previous section.

# There's more...

This section consists of additional information about the recipe in order to make you more knowledgeable about the recipe.

# See also

This section provides helpful links to other useful information for the recipe.

# Get in touch

Feedback from our readers is always welcome.

**General feedback**: 如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并发送邮件至`customercare@packtpub.com`联系我们。

**勘误**：尽管我们已竭尽全力确保内容准确性，但错误难免会发生。如果您在本书中发现错误，请向我们报告，我们将不胜感激。请访问[www.packtpub.com/support/errata](https://www.packtpub.com/support/errata)，选择您的书籍，点击“勘误提交表格”链接，并填写详细信息。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法复制，请向我们提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并附上材料链接。

**如果您有兴趣成为作者**：如果您在某个专题上有专业知识，并且有意写作或为书籍做贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评论。在阅读并使用本书后，为何不在您购买它的网站上留下评论呢？潜在的读者可以看到并使用您的客观意见来作出购买决策，我们在 Packt 可以了解您对我们产品的看法，而我们的作者可以看到您对他们书籍的反馈。谢谢！

要了解更多关于 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
