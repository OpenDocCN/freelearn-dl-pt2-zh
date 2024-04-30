# 前言

Transformers对于**自然语言理解**（**NLU**）的重要性不言而喻，NLU 是**自然语言处理**（**NLP**）的一个子集，在全球数字经济中已经成为人工智能的支柱之一。

Transformers模型标志着人工智能的新时代的开始。语言理解已成为语言建模、聊天机器人、个人助手、问答、文本摘要、语音转文本、情感分析、机器翻译等领域的支柱。我们正在见证社交网络与面对面的社交相比的扩张，电子商务与实体购物的竞争，数字新闻与传统媒体的竞争，流媒体与实体剧院的竞争，远程医生咨询与实体就诊的竞争，远程工作与现场任务的竞争以及数百个领域中类似的趋势。如果没有 AI 语言理解，社会将难以使用网络浏览器、流媒体服务以及任何涉及语言的数字活动。我们社会从物理信息到大规模数字信息的范式转变迫使人工智能进入了一个新时代。人工智能已经演化到了亿级参数模型来应对万亿字数据集的挑战。

Transformers架构既具有革命性又具有颠覆性。它打破了过去，摆脱了 RNN 和 CNN 的主导地位。 BERT 和 GPT 模型放弃了循环网络层，并用自注意力替换它们。Transformers模型胜过了 RNN 和 CNN。 2020 年代正在经历人工智能的重大变革。

Transformers编码器和解码器包含可以单独训练的注意力头，可以并行化最先进的硬件。注意力头可以在单独的 GPU 上运行，为亿级参数模型和即将推出的万亿级参数模型敞开大门。OpenAI 在拥有 10000 个 GPU 和 285000 个 CPU 核心的超级计算机上训练了一个有 1750 亿参数的 GPT-3 Transformers模型。

数据量的增加要求以规模训练 AI 模型。因此，Transformers为参数驱动 AI 开辟了新的时代。学会理解数亿个单词在句子中如何组合需要大量参数。

谷歌 BERT 和 OpenAI GPT-3 等Transformers模型已经将新兴技术推上了新的高度。Transformers可以执行它们未经过训练的数百种 NLP 任务。

Transformers还可以通过将图像嵌入为单词序列来学习图像分类和重建。本书将向您介绍最前沿的计算机视觉Transformers，如**视觉Transformers**（**ViT**），CLIP 和 DALL-E。

基础模型是完全训练的Transformers模型，可以在不进行微调的情况下执行数百种任务。在这个大规模信息时代，基础模型提供了我们所需要的工具。

想想需要多少人来控制每天在社交网络上发布的数十亿条消息的内容，以在提取其中的信息之前判断它们是否合法和道德。

想想需要多少人来翻译每天在网络上发布的数百万页。或者想象一下手动控制每分钟生成的数百万条消息需要多少人！

最后，想想需要多少人来记录每天在网络上发布的大量流媒体的转录。最后，想想替代 AI 图像字幕对那些持续出现在网上的数十亿张图片所需的人力资源。

这本书将带你从代码开发到提示设计，掌握控制 Transformer 模型行为的新 "编程" 技能。每一章都将带你从 Python、PyTorch 和 TensorFlow 的零基础开始，了解语言理解的关键方面。

你将学习到原始 Transformer、Google BERT、OpenAI GPT-3、T5 以及其他几个模型的架构。你将微调 transformer，从头开始训练模型，并学会使用强大的 API。Facebook、Google、Microsoft 和其他大型科技公司分享了大量数据集供我们探索。

你将与市场及其对语言理解的需求保持紧密联系，例如媒体、社交媒体和研究论文等领域。在数百个 AI 任务中，我们需要总结大量的研究数据，为经济的各个领域翻译文件，并为伦理和法律原因扫描所有的社交媒体帖子。

在整本书中，你将尝试使用 Python、PyTorch 和 TensorFlow 进行实践。你将了解关键的 AI 语言理解神经网络模型。然后，你将学习如何探索和实现 transformer。

在这个颠覆性的 AI 时代，你将学到成为工业 4.0 AI 专家所需的新技能。本书旨在为读者提供 Python 深度学习方面的知识和工具，以有效开发语言理解的核心内容。

# 本书适合谁阅读

本书不是 Python 编程或机器学习概念的入门。相反，它专注于机器翻译、语音转文字、文字转语音、语言建模、问答等多个 NLP 领域的深度学习。

最能从本书受益的读者有：

+   熟悉 Python 编程的深度学习和 NLP 从业者。

+   数据分析师和数据科学家想要对 AI 语言理解有个初步了解，以处理越来越多的以语言为驱动的功能。

# 本书涵盖的内容

**第一部分：Transformer 架构简介**

*第一章*，*什么是 Transformer？*，在高层次上解释了 Transformer 是什么。我们将看看 Transformer 生态系统以及基础模型的特性。本章突出了许多可用的平台以及工业 4.0 AI 专家的演变。

*第二章*，*了解 Transformer 模型的架构初步*，通过 NLP 的背景来理解 RNN、LSTM 和 CNN 深度学习架构是如何演变成打开新时代的 Transformer 架构的。我们将通过 Google Research 和 Google Brain 的作者们提出的独特 *注意力就是一切* 方法来详细了解 Transformer 的架构。我们将描述Transformers的理论。我们将用 Python 亲自动手来看看多头注意力子层是如何工作的。在本章结束时，你将了解到 Transformer 的原始架构。你将准备好在接下来的章节里探索 Transformer 的多种变体和用法。

*第三章*，*微调 BERT 模型*，在原始 Transformer 的架构基础上构建。**来自Transformers的双向编码表示**（**BERT**）向我们展示了一个感知 NLP 世界的新方式。BERT 不是分析过去的序列以预测未来的序列，而是关注整个序列！我们将首先了解 BERT 架构的关键创新，然后通过在 Google Colaboratory 笔记本中逐步微调 BERT 模型。像人类一样，BERT 可以学习任务，并且执行其他新任务而无需从头学习话题。

*第四章*，*从头开始预训练 RoBERTa 模型*，使用 Hugging Face PyTorch 模块从头构建一个 RoBERTa Transformers模型。 这个Transformers模型将类似于 BERT 和 DistilBERT。首先，我们将在一个定制的数据集上从头开始训练一个分词器。然后，训练好的Transformers模型将在一个下游掩码语言建模任务中运行。

**第 II 部分：将 Transformer 应用于自然语言理解和生成**

*第五章*，*使用 Transformer 进行下游 NLP 任务*，展示了使用下游 NLP 任务的魔力。一个预训练的Transformers模型可以被微调以解决一系列 NLP 任务，如 BoolQ、CB、MultiRC、RTE、WiC 等，在 GLUE 和 SuperGLUE 排行榜上占据主导地位。我们将介绍Transformers的评估过程、任务、数据集和指标。然后我们将使用 Hugging Face 的Transformers管道运行一些下游任务。

*第六章*，*使用 Transformer 进行机器翻译*，定义机器翻译以了解如何从人类标准到机器传导方法。然后，我们将预处理来自欧洲议会的 WMT 法英数据集。机器翻译需要精确的评估方法，在本章中，我们将探讨 BLEU 评分方法。最后，我们将使用 Trax 实现一个 Transformer 机器翻译模型。

*第七章*，*超人类 Transformers 的崛起与 GPT-3 引擎*，探讨了 OpenAI 的 GPT-2 和 GPT-3 transformers 的许多方面。我们将首先检查 OpenAI 的 GPT 模型的架构，然后解释不同的 GPT-3 引擎。接着我们将运行一个 GPT-2 345M 参数模型，并与其交互生成文本。接下来，我们将看到 GPT-3 游乐场的运作方式，然后编写一个用于 NLP 任务的 GPT-3 模型，并将结果与 GPT-2 进行比较。

*第八章*，*将 Transformer 应用于法律和金融文件以进行 AI 文本摘要*，介绍了 T5 Transformer 模型的概念和架构。我们将使用 Hugging Face 的 T5 模型来进行文档摘要。我们将让 T5 模型摘要各种文档，包括 *权利法案* 中的样本，探讨将迁移学习方法应用于 Transformers 的成功和局限性。最后，我们将使用 GPT-3 将一些公司法文本摘要成二年级水平的语言。

*第九章*，*匹配 Tokenizers 和数据集*，分析了分词器的限制，并研究了一些用于改进数据编码过程质量的方法。我们将首先构建一个 Python 程序来研究为什么一些单词会被 word2vector 分词器省略或误解。在此之后，我们将使用一个与分词器无关的方法来找出预训练分词器的限制。

我们将通过应用一些显示仍有很大改进空间的想法来改进 T5 摘要方法论。最后，我们将测试 GPT-3 的语言理解能力的限制。

*第十章*，*使用基于 BERT 的 Transformers 进行语义角色标注*，探讨了 Transformers 如何学习理解文本内容。**语义角色标注**（**SRL**）对人类来说是一个具有挑战性的练习。Transformers 可以产生令人惊讶的结果。我们将在 Google Colab 笔记本中实现由 AI 艾伦研究所设计的基于 BERT 的 Transformer 模型。我们还将使用他们的在线资源来可视化 SRL 输出。最后，我们将质疑 SRL 的范围，并理解其局限性背后的原因。

**第三部分：高级语言理解技术**

*第十一章*，*让你的数据说话：故事、问题和答案*，展示了 Transformer 如何学会推理。Transformer 必须能够理解文本、故事，并展现出推理技能。我们将看到如何通过添加 NER 和 SRL 来增强问答过程。我们将为一个问题生成器建立蓝图，该生成器可以用于训练 Transformers 或作为独立解决方案。

*第十二章*，*检测客户情绪以进行预测*，展示了Transformers如何改进情感分析。我们将使用斯坦福情感树库分析复杂句子，挑战几种Transformers模型，以理解一个序列的结构和逻辑形式。我们将看到如何使用Transformers进行预测，根据情感分析输出触发不同的动作。该章节最后通过使用 GPT-3 来解决一些边缘案例。

*第十三章*，*使用Transformers分析假新闻*，深入探讨了假新闻的热门话题以及Transformers如何帮助我们理解我们每天看到的在线内容的不同视角。每天，在线社交媒体、网站以及各种实时通讯方式发布数十亿条消息、帖子和文章。使用前几章的几种技术，我们将分析气候变化和枪支管制的辩论，以及一位前总统的推文。我们将讨论在合理怀疑的基础上确定什么可以被认为是假新闻，以及哪些新闻是主观的道德和伦理问题。

*第十四章*，*解释黑匣子Transformers模型*，通过可视化它们的活动揭开了Transformers模型的黑匣子。我们将使用 BertViz 来可视化注意力头，使用**语言可解释性工具**（**LIT**）来进行**主成分分析**（**PCA**）。最后，我们将使用 LIME 通过字典学习来可视化Transformers。

*第十五章*，*从自然语言处理到任务无关的Transformers模型*，深入研究了先进模型 Reformer 和 DeBERTa，运用 Hugging Face 运行示例。Transformers可以将图像处理为单词序列。我们还将看到不同的视觉Transformers，如 ViT、CLIP 和 DALL-E。我们将对它们进行计算机视觉任务的测试，包括生成计算机图像。

*第十六章*，*Transformers驱动联合驾驶员的出现*，探讨了工业 4.0 的成熟度。本章从使用非正式/随意英语的提示工程示例开始。接下来，我们将使用 GitHub Copilot 和 OpenAI Codex 从几行指令中创建代码。我们将看到视觉Transformers如何帮助自然语言处理Transformers可视化周围的世界。我们将创建一个基于Transformers的推荐系统，该系统可以由数字人类在您可能会进入的任何元宇宙中使用！

*附录 I*，*Transformers模型术语*，检查了一个Transformers的高层结构，从堆栈和子层到注意力头。

*附录 II*，*Transformers模型的硬件约束*，着眼于运行Transformers的 CPU 和 GPU 性能。我们将看到为什么Transformers和 GPU 以及Transformers是完美的匹配，最后进行一个测试，使用 Google Colab CPU、Google Colab 免费 GPU 和 Google Colab Pro GPU。

*附录 III*，*使用 GPT-2 进行通用文本完成*，详细解释了使用 GPT-2 进行通用文本完成的内容，从*第七章*，*GPT-3 引擎的超人类转变*。

*附录 IV*，*使用 GPT-2 进行自定义文本完成*，通过构建和训练 GPT-2 模型并使其与自定义文本交互，补充了*第七章*，*GPT-3 引擎的超人类转变*。

*附录 V*，*问题答案*，提供了每章末尾问题的答案。

# 充分利用本书

本书中的大多数程序都是 Colaboratory 笔记本。您只需要一个免费的 Google Gmail 帐户，就可以在 Google Colaboratory 的免费虚拟机上运行这些笔记本。

您需要在计算机上安装 Python 才能运行一些教育性程序。

花费必要的时间阅读*第二章*，*开始理解Transformers模型的架构* 和 *附录 I*，*Transformers模型术语*。*第二章* 包含了原始Transformers的描述，该描述是由 *附录 I*，*Transformers模型术语* 中解释的构建块构成的，这些构建块将在本书中实现。如果您觉得困难，可以从章节中提取一般的直观想法。当您在几章后对Transformers感到更加舒适时，您可以再回到这些章节。

阅读完每章后，考虑您如何为您的客户实施Transformers或使用它们提出新颖的想法来提升您的职业生涯。

请注意，我们在书的后面使用 OpenAI Codex，目前有等待名单。现在注册以避免长时间等待，请访问[`openai.com/blog/openai-codex/`](https://openai.com/blog/openai-codex/)。

## 下载示例代码文件

本书的代码包托管在 GitHub 上，地址为[`github.com/Denis2054/Transformers-for-NLP-2nd-Edition`](https://github.com/Denis2054/Transformers-for-NLP-2nd-Edition)。我们还提供了来自我们丰富书目和视频目录的其他代码包，请查看[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。

## 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的截图/图表的彩色图像。您可以在此处下载：[`static.packt-cdn.com/downloads/9781803247335_ColorImages.pdf`](https://static.packt-cdn.com/downloads/9781803247335_ColorImages.pdf)。

## 使用约定

本书中使用了几种文本约定。

`CodeInText`：指示在书中运行的句子和单词，文本中的代码单词，数据库表名，文件夹名，文件名，文件扩展名，路径名，虚拟 URL，用户输入和 Twitter 句柄。例如，“但是，如果您想要探索代码，您可以在本章的 GitHub 代码库中找到 Google Colaboratory 的 `positional_encoding.ipynb` 笔记本和 `text.txt` 文件。”

代码块设置如下：

```py
import numpy as np
from scipy.special import softmax 
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项会以粗体显示：

```py
The **black** cat sat on the couch and the **brown** dog slept on the rug. 
```

任何命令行输入或输出都按如下方式编写：

```py
vector similarity
[[0.9627094]] final positional encoding similarity 
```

**粗体**：表示新术语、重要词语或在屏幕上看到的文字，例如在菜单或对话框中，也会在文本中以此形式出现。例如：“在我们的情况下，我们正在寻找**t5-large**，一个我们可以在 Google Colaboratory 中顺利运行的 t5-large 模型。”

警告或重要提示以这种方式出现。

提示和技巧以这种方式出现。

# 联系我们

我们的读者的反馈总是受欢迎的。

**一般反馈**：发送电子邮件至`feedback@packtpub.com`，在邮件主题中提及书籍的标题。如果您对本书的任何方面有疑问，请通过`questions@packtpub.com`与我们联系。

**勘误**：尽管我们已经尽一切努力确保我们内容的准确性，但错误是不可避免的。如果您在本书中发现错误，请将此错误报告给我们。请访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击错误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上任何形式的地方发现我们作品的任何非法复制，我们将不胜感激您能提供给我们位置地址或网站名称。请通过`copyright@packtpub.com`与我们联系，并附上链接到该材料的链接。

**如果您对成为作者感兴趣**：如果您对某个您擅长的主题感兴趣，并且您有兴趣编写或为一本书做出贡献，请访问[`authors.packtpub.com`](http://authors.packtpub.com)。

# 分享您的想法

一旦您阅读完*自然语言处理的Transformers-第二版*，我们将很高兴听到您的想法！请[点击此处直达亚马逊评论页面](https://packt.link/r/1803247339)给这本书留下您的反馈。

您的审阅对我们和技术社区都很重要，并将帮助我们确保我们提供的内容质量卓越。
