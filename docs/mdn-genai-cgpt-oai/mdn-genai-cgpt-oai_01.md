# 序言

本书以介绍生成式 AI 领域开始，重点是使用机器学习算法创建新的独特数据或内容。它涵盖了生成式 AI 模型的基础知识，并解释了这些模型如何训练以生成新数据。

之后，它着重于 ChatGPT 如何提升生产力和增强创造力的具体用例。它还探讨了如何通过改进提示设计和利用零、一、和少次学习能力来充分利用 ChatGPT 互动。

本书随后对之前按领域聚类的用例进行了详细介绍：营销人员、研究人员和开发人员/数据科学家。每个领域将涵盖四个具体的用例，您可以轻松地自行复制。

然后，从个别用例开始，本书转向利用 Azure 基础设施上可用的 OpenAI 模型 API 的企业级场景。本书还将重点关注现有客户故事的端到端场景，以及负责任的 AI 影响。

最后，本书将回顾迄今讨论的主要要点，并反思生成式 AI 领域接下来的发展方向。

通过本书，您将掌握深入研究生成式 AI 领域并开始在自己的项目中使用 ChatGPT 和 OpenAI 模型 API 的知识。

# 本书适合谁

本书面向广泛的读者群体。它适用于对使用 ChatGPT 和 OpenAI 提高日常生产力以及深入了解 ChatGPT 背后的技术和模型架构感兴趣的一般用户。它也适用于希望深入了解 ChatGPT 和 OpenAI 模型在现实世界应用中的商业用户，并了解它们如何赋能其组织。本书还适用于希望深入了解 ChatGPT 和 OpenAI 模型如何提升其 ML 模型和代码的数据科学家和开发人员，以及希望深入了解其领域中 OpenAI 和 ChatGPT 用例的营销人员和研究人员。

由于本书提供了 OpenAI 模型背后的技术和生成式 AI 主要概念的理论概述，因此阅读本书并无特殊要求。如果您还对实施实际用例感兴趣，我们提供了端到端的解决方案和相关 Python 代码，以及逐步实施说明。

# 本书涵盖了什么内容

*第一章*，*生成式 AI 简介*，提供了生成式 AI 领域的概述，重点是使用机器学习算法创建新的独特数据或内容。它涵盖了生成式 AI 模型的基础知识，并解释了这些模型如何训练以生成新数据。该章节还着重介绍了生成式 AI 在各个领域的应用，如图像合成、文本生成和音乐创作，突出了生成式 AI 在革新各行业中的潜力。

*第二章*，*OpenAI 和 ChatGPT – 超越市场炒作*，概述了 OpenAI 及其最显著的发展 ChatGPT，重点介绍其历史、技术和能力。本章还关注了 ChatGPT 如何在各行业和应用中使用，以改善沟通和自动化流程，最终如何影响技术领域及其他领域。

*第三章*，*熟悉 ChatGPT*，指导您如何设置 ChatGPT 账户并开始使用该服务。还将介绍如何与 Web UI 交互，如何根据主题组织聊天，以及如何构建对话。

*第四章*，*理解提示设计*，着重介绍了提示设计的重要性作为提高模型准确性的技术。实际上，提示对模型生成的输出有很大影响。一个设计良好的提示可以帮助引导模型生成相关和准确的输出，而一个设计不当的提示可能是无关或令人困惑的。最后，还重要的是在提示中实施道德考虑，以防止模型生成有害内容。

*第五章*，*ChatGPT 提升日常工作效率*，介绍了 ChatGPT 可以为普通用户在日常生活中做的主要活动，提高用户的生产力。本章将重点介绍写作辅助、决策制定、创意灵感等具体示例，让您可以自己尝试。

*第六章*，*与 ChatGPT 共同开发未来*，着重介绍了开发人员如何利用 ChatGPT。本章将重点关注 ChatGPT 在这一领域可以解决的主要用例，包括代码审查和优化、文档生成和代码生成。本章将提供示例，并让您尝试自己的提示。

*第七章*，*ChatGPT 精通营销*，着重介绍了营销人员如何利用 ChatGPT。本章将重点关注 ChatGPT 在这一领域可以解决的主要用例，包括 A/B 测试、关键词定位建议和社交媒体情感分析。本章将提供示例，并让您尝试自己的提示。

*第八章*，*ChatGPT 改变研究方式*，着重介绍了研究人员如何利用 ChatGPT。本章将重点关注 ChatGPT 在这一领域可以解决的主要用例，包括文献综述、实验设计和参考文献生成。本章将提供示例，并让您尝试自己的提示。

*第九章*，*OpenAI 和 ChatGPT 企业版 – 介绍 Azure OpenAI*，着重介绍了 OpenAI 模型在企业级应用中的应用，介绍了 Azure OpenAI 服务。我们将概述 Azure OpenAI 服务的模型 API 以及如何将其嵌入自定义代码中。我们还将关注模型参数、配置和微调。最后，我们将对负责任人工智能的主题进行一些考虑，以确保您的人工智能系统符合道德标准。

*第十章*，*企业的热门用例*，以当前市场上企业正在开发的使用 Azure OpenAI 的最热门用例概述开始。我们将探讨项目的具体示例，如智能搜索引擎、人工智能助手和报告生成器。最后，我们将专注于特定行业的端到端生产项目。

*第十一章*，*结语和最终思考*，以对前几章中探讨的最热门用例进行简短回顾开始。然后，我们将转向一些关于生成式人工智能对行业和日常生活影响的考虑。我们还将了解道德考虑和负责任人工智能在项目设计阶段的作用。本章将以一些关于生成式人工智能未来发展的最终思考结束，GPT-4 即将推出。

# 要充分利用本书

这里是您需要准备的清单：

| **书中涵盖的软件/硬件** | **系统要求** |
| --- | --- |
| Python 3.7.1 或更高版本 | Windows、macOS 或 Linux |
| Streamlit | Windows、macOS 或 Linux |
| LangChain | Windows、macOS 或 Linux |
| OpenAI 模型 API | OpenAI 账户 |
| Azure OpenAI 服务 | 已启用 Azure OpenAI 的 Azure 订阅 |

与往常一样，您可以在书的 GitHub 存储库中找到所有章节中使用的提示：[`github.com/PacktPublishing/Modern-Generative-AI-with-ChatGPT-and-OpenAI-Models/tree/main/Chapter%204%20-%20Prompt%20design`](https://github.com/PacktPublishing/Modern-Generative-AI-with-ChatGPT-and-OpenAI-Models/tree/main/Chapter%204%20-%20Prompt%20design)

**如果您使用本书的数字版本，我们建议您自己输入代码或从书的 GitHub 存储库中访问代码（链接在下一节中提供）。这样做将有助于避免与复制粘贴代码相关的任何潜在错误**。

# 下载示例代码文件

您可以从 GitHub 下载本书的示例代码文件：[`github.com/PacktPublishing/Modern-Generative-AI-with-ChatGPT-and-OpenAI-Models`](https://github.com/PacktPublishing/Modern-Generative-AI-with-ChatGPT-and-OpenAI-Models)。如果代码有更新，将在 GitHub 存储库中更新。

我们还提供来自我们丰富书籍和视频目录的其他代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。快来看看吧！

# 下载彩色图片

我们还提供一份 PDF 文件，其中包含本书中使用的屏幕截图和图表的彩色图片。您可以在此处下载：[`packt.link/YFTZk`](https://packt.link/YFTZk)。

# 使用的约定

本书中使用了许多文本约定。

`文本中的代码`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这里有一个例子：“将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为系统中的另一个磁盘。”

代码块设置如下：

```py
query = st.text_area("Ask a question about the document")
if query:
    docs = faiss_index.similarity_search(query, k=1)
    button = st.button("Submit")
    if button:
        st.write(get_answer(faiss_index, query))
```

任何命令���输入或输出都将如下所示：

```py
pip install --upgrade openai
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词以**粗体**显示。这里有一个例子：“您可以选择通过选择**本地文件**或**Azure blob 或其他共享** **网络位置**来上传文件。”

提示或重要说明

显示如此。

