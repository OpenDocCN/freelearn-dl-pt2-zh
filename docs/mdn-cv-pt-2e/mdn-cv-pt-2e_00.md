# 前言

**人工智能** (**AI**) 已经到来，并成为推动日常应用的强大力量。 就像火的发现/发明，轮子，石油，电力和电子技术一样，AI 正在以我们难以想象的方式重塑我们的世界。 AI 历来是一个计算机科学的小众学科，只有少数实验室提供。 但由于出色理论的爆炸，计算能力的增加和数据的可用性，这一领域在 21 世纪初开始呈指数增长，并显示出不会很快放缓的迹象。

AI 一再证明，只要给定合适的算法和足够的数据量，它可以几乎不需要人为干预地学习任务，并产生可以与人类判断相匹敌甚至超越的结果。 无论您是刚开始学习或是驾驭大型组织的资深人士，都有充分的理由了解 AI 的工作原理。 **神经网络** (**NNs**) 是适应广泛应用的最灵活的 AI 算法之一，包括结构化数据，文本和视觉领域。

本书从 NN 的基础知识开始，涵盖了超过 40 个使用**PyTorch**的计算机视觉应用。 通过掌握这些应用，您将能够为各种领域（如汽车，安全，金融后勤，医疗保健等）的各种用例构建 NN，具备必要的技能不仅能实施最先进的解决方案，还能创新并开发解决更多现实世界挑战的新应用。

最终，本书旨在成为学术学习和实际应用之间的桥梁，使您能够自信前行，并在职业生涯中做出重要贡献。

# 本书适合的读者

本书适合于刚接触 PyTorch 和中级机器学习实践者，他们希望通过深度学习和 PyTorch 熟悉 CV 技术。 刚开始接触 NN 的人士也会发现本书有用。 您只需具备 Python 编程语言和机器学习的基础知识即可开始本书。

# 本书内容涵盖了什么

*第一章*，*人工神经网络基础*，为您详细介绍了 NN 的工作原理。 您将首先学习与 NN 相关的关键术语。 接下来，您将了解构建块的工作细节，并在玩具数据集上从头构建 NN。 在本章结束时，您将对 NN 的工作原理感到自信。

*第二章*，*PyTorch 基础*，介绍了如何使用 PyTorch。 您将了解创建和操作张量对象的方法，然后学习使用 PyTorch 构建神经网络模型的不同方法。 您将继续使用玩具数据集，以便了解与 PyTorch 的具体工作方式。

*第三章*，*使用 PyTorch 构建深度神经网络*，结合前几章的内容，深入理解各种神经网络超参数对模型准确性的影响。通过本章的学习，你将自信地在真实数据集上操作神经网络。

*第四章*，*引入卷积神经网络*，详细介绍了使用传统神经网络的挑战，以及卷积神经网络（CNNs）如何克服传统神经网络的各种局限性的原因。你将深入了解 CNN 的工作细节和其中的各个组件。接下来，你将学习在处理图像时的最佳实践。在本章中，你将开始处理真实世界的图像，并学习 CNN 如何帮助进行图像分类的复杂性。

*第五章*，*图像分类的迁移学习*，让你在实际中解决图像分类问题。你将学习多种迁移学习架构，了解它们如何显著提高图像分类的准确性。接下来，你将利用迁移学习实现面部关键点检测和年龄性别估计的使用案例。

*第六章*，*图像分类的实际方面*，深入探讨了在构建和部署图像分类模型时需要注意的实际方面。你将实际看到如何利用数据增强和批归一化在真实数据上带来优势。此外，你将学习类激活图如何帮助解释 CNN 模型为什么会预测某个结果的原因。通过本章的学习，你将能够自信地解决大多数图像分类问题，并在自定义数据集上利用前三章讨论的模型。

*第七章*，*物体检测基础*，为物体检测奠定了基础，你将了解用于构建物体检测模型的各种技术。接下来，通过一个使用案例，你将学习基于区域提议的物体检测技术，实现一个定位图像中卡车和公共汽车的模型。

*第八章*，*高级物体检测*，让你了解了基于区域提议的架构的局限性。接下来，你将学习更先进的架构，如 YOLO 和 SSD，它们解决了基于区域提议的架构的问题。你将在同一数据集（卡车与公共汽车检测）上实现所有这些架构，以便对比每种架构的工作原理。

*第九章*，*图像分割*，在前几章的基础上建立，并将帮助你构建能准确定位各种类别对象及其实例在图像中位置的模型。你将在道路图像和普通家居图像上实现使用案例。本章结束时，你将能够自信地解决任何图像分类和对象检测/分割问题，并通过使用 PyTorch 构建模型来解决这些问题。

*第十章*，*目标检测和分割的应用*，总结了我们在所有前几章中学到的内容，并开始在几行代码中实现目标检测和分割，并实现模型来执行人群计数和图像着色。接下来，你将学习在真实数据集上进行 3D 目标检测。最后，你将学习如何在视频上执行动作识别。

*第十一章*，*自编码器和图像处理*，为修改图像打下基础。你将从学习各种自编码器开始，这些自编码器有助于压缩图像并生成新颖图像。接下来，你将学习对抗攻击，这些攻击能欺骗模型，在实施神经风格转换之前。最后，你将实现一个自编码器来生成深度伪造图像。

*第十二章*，*使用 GAN 生成图像*，首先深入介绍了 GAN 的工作原理。接下来，你将实现虚假面部图像生成，并使用 GAN 生成感兴趣的图像。

*第十三章*，*高级 GAN 用于图像操作*，将图像操作推向了新的高度。你将使用 GAN 来将对象从一类转换到另一类，从草图生成图像，并操作自定义图像，以便按特定风格生成图像。本章结束时，你将能够自信地使用自编码器和 GAN 的组合进行图像操作。

*第十四章*，*结合计算机视觉和强化学习*，首先介绍了强化学习（RL）的术语和为状态分配价值的方式。当你学习深度 Q 学习时，你将了解到如何将 RL 和神经网络结合起来。利用这些知识，你将实现一个代理来玩乒乓球游戏，以及一个实现自动驾驶汽车的代理。

*第十五章*，*结合计算机视觉和 NLP 技术*，详细介绍了变压器的工作原理，你将利用它实现诸如图像分类、手写识别、护照图像中的键-值提取以及最后的图像视觉问答等应用。在这个过程中，你将学习多种自定义/利用变压器架构的方法。

*Chapter 16*, *Foundation Models in Computer Vision,* starts by strengthening your understanding of combining image and text using CLIP model. Next, you will discuss the Segment Anything Model (SAM), which helps with a variety of tasks – segmentation, recognition, and tracking without any training. Finally, you will understand the working details of diffusion models before you learn the importance of prompt engineering and the impact of bigger pre-trained models like SDXL.

*Chapter 17*, *Applications of Stable Diffusion,* extends what you learned in the previous chapters by walking you through how a variety of Stable Diffusion applications (image in-painting, ControlNet, DepthNet, SDXL Turbo, and text-to-video) are trained and then walking you through leveraging different models to perform different tasks.

*Chapter 18*, *Moving a Model to Production*, describes the best practices for moving a model to production. You will first learn about deploying a model on a local server before moving it to the AWS public cloud. Next, you will learn about the impact of half-precision on latency, and finally, you will learn about leveraging vector stores (for instance, FAISS) and identifying data drift once a model is moved to production.

随着领域的发展，我们将定期向 GitHub 存储库添加有价值的补充内容。请检查每个章节目录中的`supplementary_sections`文件夹以获取新的有用内容。

# 要充分利用本书

| **Software/hardware covered in the book** | **OS requirements** |
| --- | --- |
| Minimum 128 GB storageMinimum 8 GB RAMIntel i5 processor or betterNVIDIA 8+ GB graphics card – GTX1070 or betterMinimum 50 Mbps internet speed | Windows, Linux, and macOS |
| Python 3.6 and above | Windows, Linux, and macOS |
| PyTorch 2.1 | Windows, Linux, and macOS |
| Google Colab (can run in any browser) | Windows, Linux, and macOS |

请注意，本书中的几乎所有代码都可以通过点击 GitHub 上每个笔记本章节的**Open Colab**按钮在 Google Colab 中运行。

如果您使用本书的数字版本，我们建议您自行输入代码或通过 GitHub 存储库访问代码（链接在下一部分提供）。这样做可以帮助您避免与复制粘贴代码相关的潜在错误。

## 下载示例代码文件

本书的代码包托管在 GitHub 上，链接为：[`github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch-2E`](https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch-2E)。我们还提供了来自丰富书籍和视频目录中的其他代码包。请查看！

## 下载彩色图像

我们还提供了一份包含本书中使用的屏幕截图/图表的彩色图像的 PDF 文件。您可以在此处下载：[`packt.link/gbp/9781803231334`](https://packt.link/gbp/9781803231334)。

## 使用约定

本书采用了许多文本约定。

`CodeInText`：表示文本中的代码词语、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。以下是一个例子：“我们正在创建`FMNISTDataset`类的`val`对象，除了我们之前看到的`train`对象。”

代码块设置如下：

```py
# Crop image
img = img[50:250,40:240]
# Convert image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Show image
plt.imshow(img_gray, cmap='gray') 
```

当我们希望引起您对代码块特定部分的注意时，相关行或项将以粗体显示：

```py
**def****accuracy****(****x, y, model****):**
    **model.****eval****()** # <- let's wait till we get to dropout section
    # get the prediction matrix for a tensor of `x` images
    **prediction = model(x)**
    # compute if the location of maximum in each row coincides
    # with ground truth
    **max_values, argmaxes = prediction.****max****(-****1****)**
    **is_correct = argmaxes == y**
    **return** **is_correct.cpu().numpy().tolist()** 
```

任何命令行输入或输出均如下所示：

```py
$ python3 -m venv fastapi-venv
$ source fastapi-env/bin/activate 
```

**粗体**：表示一个新术语，一个重要词或您在屏幕上看到的单词。例如，菜单或对话框中的词语会出现在文本中，如此处所示。以下是一个例子：“我们将使用梯度下降（在前馈传递后）逐个**批次**进行，直到我们在**一个训练轮次**内用尽所有数据点。”

警告或重要说明如下所示。

提示和技巧如下所示。

# 与我们联系

我们始终欢迎读者的反馈。

**一般反馈**：电子邮件 `feedback@packtpub.com`，并在您的消息主题中提及书名。如果您对本书的任何方面有疑问，请发送电子邮件至 `questions@packtpub.com` 联系我们。

**勘误**：尽管我们尽了最大努力确保内容的准确性，但错误难免会发生。如果您在本书中发现错误，我们将不胜感激您向我们报告。请访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法副本，请向我们提供位置地址或网站名称将不胜感激。请通过 `copyright@packtpub.com` 提供链接至该材料的链接。

**如果您有兴趣成为作者**：如果您在某个专题上有专业知识，并且有兴趣撰写或为书籍做贡献，请访问 [`authors.packtpub.com`](http://authors.packtpub.com)。

# 分享您的想法

一旦您阅读完*Modern Computer Vision with PyTorch, Second Edition*，我们很想听听您的想法！请[点击此处直接访问亚马逊评论](https://packt.link/r/1803231335)页面，与我们分享您的反馈。

您的评论对我们和技术社区都非常重要，将帮助我们确保提供优质内容。

# 下载本书的免费 PDF 副本

感谢购买本书！

您喜欢随时随地阅读，但无法随身携带印刷书籍吗？

您购买的电子书与您选择的设备不兼容吗？

别担心，现在每本 Packt 书籍都可以免费获取不受 DRM 限制的 PDF 版本。

无论何时何地，任何设备上都能阅读。直接从您喜爱的技术书籍中搜索、复制和粘贴代码到您的应用程序中。

福利不止于此，您还可以在每天的电子邮箱中获取独家折扣、新闻简报和精彩的免费内容。

按照以下简单步骤获取这些福利：

1.  扫描 QR 码或访问以下链接：

![](img/B18457_Free_PDF_QR.png)

[`packt.link/free-ebook/9781803231334`](https://packt.link/free-ebook/9781803231334)

1.  提交您的购买证明。

1.  这就是！我们将免费的 PDF 和其他福利直接发送到您的电子邮箱。
