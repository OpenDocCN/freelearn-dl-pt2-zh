# 序言

> “想象力比知识更重要。”
> 
> – 阿尔伯特·爱因斯坦，《爱因斯坦关于宇宙宗教和其他见解与格言》（2009）

在本书中，我们将探索生成式人工智能，这是一种使用先进的机器学习算法生成合成（但惊人逼真）数据的尖端技术。生成模型一直以来都引起了跨领域研究人员的兴趣。随着机器学习和更具体地说是深度学习领域的最新进展，生成式建模在研究作品数量和它们在不同领域的应用中迅速增长。从艺术作品和音乐作曲到合成医学数据集，生成建模正在推动想象力和智能的界限。理解、实现和利用这些方法所需的思考和努力量简直令人惊叹。一些较新的方法（如 GANs）非常强大，但难以控制，使得整个学习过程既令人兴奋又令人沮丧。

*使用 Python 和 TensorFlow 2 进行生成式人工智能* 是我们作者和 *Packt Publishing* 的才华横溢团队数小时辛勤工作的结果，帮助你理解这个生成式建模领域的 *深度*、*广度* 和 *狂野* 空间。本书的目标是成为生成式建模空间的万花筒，并涵盖广泛的主题。本书带你走向旅程，在这个过程中，你不仅仅是读懂理论和了解基础知识，还通过示例发现了这些模型的潜力和影响。我们将使用各种开源技术来实现这些模型——Python 编程语言、用于深度神经网络开发的 TensorFlow 2 库，以及云计算资源，如 Google Colab 和 Kubeflow 项目。

对本书中各种主题、模型、架构和示例的理解将帮助你轻松探索更复杂的主题和尖端研究。

# 本书适合对象

*使用 Python 和 TensorFlow 2 进行生成式人工智能* 面向数据科学家、机器学习工程师、研究人员和对生成式建模以及将最先进的架构应用于真实数据集感兴趣的开发者。这本书也适合那些具有中级深度学习相关技能的 TensorFlow 初学者，他们希望扩展自己的知识基础。

开始阅读本书只需要基本的 Python 和深度学习技能。

# 本书内容包括

*第一章*，*生成式人工智能简介：从模型中“提取”数据*，介绍了生成式人工智能领域，从概率论基础到最近的这些方法的应用产品。

*第二章*，*建立一个 TensorFlow 实验室*，描述了如何使用开源工具——Python、Docker、Kubernetes 和 Kubeflow——为使用 TensorFlow 开发生成式人工智能模型设置计算环境，以便在云中运行可扩展的代码实验室。

*第三章*，*深度神经网络的构建模块*，介绍了深度神经网络的基础概念，这些概念将在本卷的其余部分中被利用——它们是如何受生物研究启发的，研究人员在开发越来越大规模和复杂模型时克服了哪些挑战，以及网络架构、优化器和正则化器的各种构建模块，这些构建模块在本书其余部分的生成式人工智能示例中被利用。

*第四章*，*教网络生成数字*，演示了如何实现深度置信网络，这是一种突破性的神经网络架构，通过生成式人工智能方法在分类手写数字图像方面取得了最先进的结果，这种方法教会网络在学习对其进行分类之前生成图像。

*第五章*，*使用 VAE 用神经网络绘制图片*，描述了变分自动编码器（VAEs），这是从深度置信网络发展而来的一种先进技术，通过巧妙地使用贝叶斯统计学中的客观函数来创建复杂对象的更清晰图像。读者将实现一个基本和高级的 VAE，后者利用了逆自回归流（IAF），这是一种递归变换，可以将随机数映射到复杂的数据分布以创建引人注目的合成图像。

*第六章*，*使用 GAN 进行图像生成*，介绍了生成对抗网络，或 GANs，作为强大的生成建模深度学习架构。从 GANs 的基本构建模块和其他基本概念开始，本章介绍了许多 GAN 架构以及它们如何用于从随机噪声生成高分辨率图像。

*第七章*，*使用 GAN 进行风格转移*，专注于生成建模的创造性应用，特别是称为风格转移的 GAN。应用例如将黑白图像转换为彩色图像，航拍地图转换为类似谷歌地图的输出，以及去除背景，都可以通过风格转移实现。我们涵盖了许多成对和非成对的架构，如 pix2pix 和 CycleGAN。

*第八章*，*使用 GAN 进行深度伪造*，介绍了 GAN 的一个有趣且有争议的应用，称为深度伪造。该章节讨论了深度伪造的基本构建模块，例如特征和不同的操作模式，以及一些关键架构。它还包括了一些实际示例，以基于所涵盖的关键概念生成虚假照片和视频，这样读者就可以创建自己的深度伪造流水线。

*第九章*，*文本生成方法的兴起*，介绍了与文本生成任务相关的概念和技术。我们从深度学习模型中不同的文本向量表示方式入手，讲述了语言生成的基本知识。然后我们介绍了不同的架构选择和解码机制，以实现高质量的输出。本章为后续更复杂的文本生成方法奠定了基础。

*第十章*，*NLP 2.0：使用 Transformer 生成文本*，介绍了 NLP 领域最新最先进的技术，重点介绍了一些基于 Transformer 架构（如 GPT-x）的最先进的文本生成能力，以及它们如何彻底改变了语言生成和 NLP 领域。

*第十一章*，*使用生成模型创作音乐*，介绍了使用生成模型创作音乐。这是一种有趣但具有挑战性的生成模型应用，涉及理解与音乐相关的许多细微差别和概念。本章涵盖了许多不同的生成音乐的方法，从基本的 LSTMs 到简单的 GANs，最终到用于多声部音乐生成的 MuseGAN。

*第十二章*，*使用生成式 AI 玩游戏：GAIL*，描述了生成式 AI 和强化学习之间的联系，强化学习是一种机器学习的分支，教授“代理”在执行指定任务时在真实或虚拟“环境”中导航。通过 GAN 和强化学习之间的联系，读者将通过模仿跳跃运动的专家示例，教会一个跳跃形象在 3D 环境中导航。

*第十三章*，*生成式 AI 的新兴应用*，描述了最近在生成式 AI 领域的研究，涵盖了生物技术、流体力学、视频和文本合成等各个方面。

# 要最大限度地发挥本书的效益

要跟上本书的代码，请推荐以下要求：

+   硬件（用于本地计算）：

    +   128GB 硬盘

    +   8GB 内存

    +   Intel Core i5 处理器或更高版本

    +   NVIDIA 8GB 及以上显卡（GTX1070 及以上）

+   软件：

    +   Python 3.6 及以上版本

    +   TensorFlow 2.x

    +   Chrome/Safari/Firefox 浏览器（如果在云中训练，则通过 Google Colab 或 Kubeflow 直接执行代码）

## 下载示例代码文件

本书的代码包托管在 GitHub 上，网址为 [`github.com/PacktPublishing/Hands-On-Generative-AI-with-Python-and-TensorFlow-2`](https://github.com/PacktPublishing/Hands-On-Generative-AI-with-Python-and-TensorFlow-2)。我们还有其他书籍和视频的代码包，请访问[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。一并查看吧！

## 下载彩色图片

我们还提供了该书中所使用的屏幕截图/图表的彩色图像的 PDF 文件。您可以在此处下载：[`static.packt-cdn.com/downloads/9781800200883_ColorImages.pdf`](https://static.packt-cdn.com/downloads/9781800200883_ColorImages.pdf)。

## 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词，数据库表名，文件夹名，文件名，文件扩展名，路径名，虚拟 URL，用户输入和 Twitter 用户名。例如："我们可以使用`show_examples()`函数直观地绘制一些示例。"

一段代码块设置如下：  

```py
def cd_update(self, x):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        h_sample = self.sample_h(x)
        for step in range(self.cd_steps):
            v_sample = tf.constant(self.sample_v(h_sample))
            h_sample = self.sample_h(v_sample) 
```

当我们希望引起您对代码块特定部分的注意时，相关行或项目将用粗体显示：

```py
def cd_update(self, x):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        h_sample = self.sample_h(x)
        **for** **step** **in****range****(self.cd_steps):**
            v_sample = tf.constant(self.sample_v(h_sample))
            h_sample = self.sample_h(v_sample) 
```

任何命令行输入或输出如下所示：

```py
pip install tensorflow-datasets 
```

**粗体**：表示一个新术语，一个重要词汇，或者您在屏幕上看到的单词，在菜单或对话框中也会以这种方式呈现在文本中。例如："从**管理**面板中选择**系统信息**"。

警告或重要说明会出现在此处。

提示和技巧会出现在此处。

# 联系我们

我们的读者的反馈总是受欢迎的。

**一般反馈**：如果您对本书的任何方面有疑问，请在消息主题中提及书名，发送电子邮件至`customercare@packtpub.com`。

**勘误**:尽管我们已尽一切努力确保内容的准确性，但错误是无法避免的。如果您在本书中发现了错误，我们将不胜感激。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)，选择您的书籍，点击**勘误提交表单**链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何非法复制形式，我们将不胜感激，如果您提供给我们位置地址或网站名称。请联系我们，发送邮件至`copyright@packtpub.com`并附上链接到该材料的链接。

**如果您有兴趣成为作者**：如果您在某个专题上有专业知识，并且有兴趣撰写或为一本书作出贡献，请访问[`authors.packtpub.com`](http://authors.packtpub.com)。

## 评论

请留下评论。一旦您阅读并使用了本书，为什么不在您购买的网站上留下评论呢？潜在读者可以看到并使用您的客观意见来做出购买决定，我们在 Packt 可以了解到您对我们产品的看法，我们的作者也能看到您对他们书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问[packtpub.com](http://packtpub.com)。
