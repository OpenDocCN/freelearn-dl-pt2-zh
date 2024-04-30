第三章：GPT-3 与编程

几乎所有 GPT-3 的 NLP 功能都是使用 Python 编程语言创建的。但为了实现更广泛的可访问性，该 API 提供了对所有主要编程语言的预构建支持，因此用户可以选择自己喜欢的编程语言构建基于 GPT-3 的应用程序。

在本节中，我们将通过使用不同的编程语言复制一个示例来说明这是如何工作的。

提示：在每个特定于语言的章节中，我们假设您对所讨论的编程语言有基本的了解。如果您没有，您可以安全地跳过该部分。

如何使用 Python 与 OpenAI API？

Python 是数据科学和机器学习任务中最流行的语言。与传统的数据科学编程语言如 R 和 Stata 相比，Python 因其可扩展性和与数据库的良好集成而脱颖而出。它被广泛使用，并且有着发展壮大的开发者社区，使其生态系统保持最新。Python 易于学习，并且附带有像 Numpy 和 Pandas 这样的有用的数据科学库。

您可以使用名为[Chronology](https://github.com/OthersideAI/chronology)的库将 GPT-3 与 Python 配对，该库提供了一个简单直观的接口。Chronology 可以缓解每次都从头开始编写所有代码的单调工作。它的功能包括：

●        它异步调用 OpenAI API，允许您同时生成多个提示完成。

●        您可以轻松地创建和修改训练提示；例如，修改不同示例使用的训练提示相当简单。

●        它允许您通过将一个提示的输出插入到另一个提示中来链接提示。

Chronology 托管在 PyPI 上，支持 Python 3.6 及以上版本。要安装库，您可以运行以下命令：

（基础）PS D：\GPT-3 Python> pip install chronological

通过 PyPI 安装 Python 库后，让我们看一个示例，说明如何使用 GPT-3 来以二年级阅读水平总结给定的文本文档。我们会展示如何调用 API，将训练提示作为请求发送，以及获取汇总的完成作为输出。我们已经将代码发布在[Github 存储库](https://github.com/Shubhamsaboo/kairos_gpt3/tree/master/Programming_with_GPT-3/GPT-3_Python)中。

在本示例中，我们将使用以下训练提示：

我的二年级学生问我这段话是什么意思：

"""

橄榄油是从橄榄（欧洲油橄榄的果实；橄榄科）中获取的液态脂肪...

"""

我用浅显易懂的语言为他重新表述了一遍：

"""

首先，导入以下依赖项：

# 导入依赖项

from chronological import read_prompt, cleaned_completion, main

现在我们可以创建一个函数，读取训练提示并提供完成的输出。我们将此函数设置为异步的，这样我们就可以进行并行函数调用。我们将使用以下配置进行 API 参数设置：

●        最大 token 数=100

●        执行引擎="Davinci"

●        温度=0.5

●        Top-p=1

●        频率惩罚 = 0.2

●        停止序列 = ["\n\n"]

# 接受训练提示并返回完成的响应

async def summarization_example():

# 接受一个文本文件（summarize_for_a_2nd_grader）作为输入提示

prompt_summarize = read_prompt('summarize_for_a_2nd_grader')

# 调用 completion 方法以及特定的 GPT-3 参数

completion_summarize = await cleaned_completion(prompt_summarize, max_tokens=100, engine="davinci", temperature=0.5, top_p=1, frequency_penalty=0.2, stop=["\n\n"])

# 返回完成的响应

返回完成的摘要

现在我们可以创建一个异步工作流程，使用库提供的 ‘main’ 函数调用该工作流程，并在控制台中打印输出：

# 设计端到端的异步工作流程，能够并行运行多个提示。

异步函数工作流程：

# 调用摘要函数进行异步调用

text_summ_example = await summarization_example()

# 在控制台中打印结果

print('-------------------------')

print('基本示例响应: {0}'.format(text_summ_example))

print('-------------------------')

# 使用主函数调用 Chronology 来运行异步工作流

main(workflow)

将其保存为名为 ‘text_summarization.py’ 的 Python 脚本，并从终端运行以生成输出。您可以从根目录运行以下命令：

(base) PS D:\GPT-3 Python> python text_summarization.py

执行脚本后，您的控制台应该打印出以下提示的摘要：

-------------------------

基本示例响应: 橄榄油是一种来自橄榄的液态脂肪。橄榄生长在一种叫做橄榄树的树上。橄榄树是地中海最常见的树。人们用这种油来烹饪，放在沙拉上，还用作灯的燃料。

-------------------------

如果您不熟悉 Python 并且想要在不编写代码的情况下链接不同的提示，您可以使用基于 [Chronology library](https://github.com/OthersideAI/chronology-ui) 构建的 [无代码界面](https://chronology-ui.vercel.app/) 来使用拖放创建提示工作流程。查看我们的 GitHub [仓库](https://github.com/Shubhamsaboo/kairos_gpt3/tree/master/Programming_with_GPT-3/GPT-3_Python) 获取更多关于如何使用 Python 编程与 GPT-3 交互的示例。

如何使用 Go 来调用 OpenAI API？

Go 是一个开源的编程语言，融合了其他语言的元素，创建了一个功能强大、高效且用户友好的工具。许多开发人员将其称为 C 的现代版本。

Go 语言是构建需要高安全性、高速度和高可扩展性的项目的首选语言。这使得它成为金融科技行业许多项目的吸引人选项。Go 语言的主要特点如下：

●        易用性

●        现代生产力

●        高级效率的静态类型

●        网络高级性能

●        充分利用多核能力

如果你完全不懂 Go 并且想试一试，你可以[查阅文档](https://golang.org/doc/install)开始入门。

安装完成并且了解了 Go 编程的基础知识后，你可以按照以下步骤使用[Go 的 GPT-3 API 包装器](https://github.com/sashabaranov/go-gpt3)。要了解更多有关创建 Go 模块的信息，请参阅[本教程](https://golang.org/doc/tutorial/create-module)。

首先，您将创建一个模块来跟踪和导入代码依赖项。使用以下命令创建并初始化“gogpt”模块：

D:\GPT-3 Go> go mod init gogpt

创建“gogpt”模块之后，让我们将其指向[此 GitHub 存储库](http://github.com/sashabaranov/go-gpt3)，以下载处理 API 的必要依赖项和包。使用以下命令：

D:\GPT-3 Go> go get github.com/sashabaranov/go-gpt3

go get：添加了 github.com/sashabaranov/go-gpt3 v0.0.0-20210606183212-2be4a268a894

我们将使用与上一节相同的文本摘要示例。（你可以在以下[存储库](https://github.com/Shubhamsaboo/kairos_gpt3/tree/master/Programming_with_GPT-3/GPT-3_Go)中找到所有代码。）

让我们首先导入必要的依赖项和包：

# 调用 package main

package main

# 导入依赖项

import (

"fmt"

"io/ioutil"

"context"

gogpt "github.com/sashabaranov/go-gpt3"

)

Go 编程将源文件组织成系统目录，称为包，这使得在 Go 应用程序中跨应用程序重用代码更加容易。在代码的第一行中，我们调用了包“main”，并告诉 Go 编译器，这个包应该编译为可执行程序，而不是共享库。

注意：在 Go 语言中，你可以创建一个可复用的共享库作为包，创建一个可执行程序的“main”包。包内的“main”函数作为程序的入口点。

现在，你将创建一个主函数，用于托管读取训练提示和提供完成输出的整个逻辑。使用以下配置设置 API 参数：

●        最大标记=100

●        执行引擎="davinci"

●        温度=0.5

●        Top-p=1

●        频率惩罚=0.2

●        结束序列= ["\n\n"]

func main() {

c := gogpt.NewClient("OPENAI-API-KEY")

ctx := context.Background()

prompt, err := ioutil.ReadFile("prompts/summarize_for_a_2nd_grader.txt")

req := gogpt.CompletionRequest{

MaxTokens: 100，

温度：0.5，

TopP: 1.0,

Stop: []string{"\n\n"},

频率惩罚：0.2，

提示：string(prompt),

}

resp, err := c.CreateCompletion(ctx, "davinci", req)

如果出现错误 {

返回

}

fmt.Println("-------------------------")

fmt.Println(resp.Choices[0].Text)

fmt.Println("-------------------------")

}

此代码执行以下任务:

1.  通过提供 API 令牌设置新的 API 客户端，然后将其留在后台运行。

1.  以文本文件的形式读取提示“”从 prompts 文件夹。

1.  通过提供训练提示并指定值 API 参数（如温度、顶部-p、停止序列等）创建完成请求。

1.  调用 create completion 函数并提供 API 客户端、完成请求和执行引擎。

1.  生成一个以完成形式的响应，最终在控制台朝着结尾打印。

然后，您可以将代码文件保存为'text_summarization.go'并从终端运行它以生成输出。 使用以下命令从您的根文件夹运行文件:

(base) PS D:\GPT-3 Go> go run text_summarization.go

一旦执行文件，您的控制台将打印以下输出:

-------------------------

橄榄油是一种来自橄榄的液态脂肪。 橄榄生长在一种被称为橄榄树的树上。 橄榄树是地中海最常见的树木。 人们用这种油来烹饪、放在沙拉上，还用作灯的燃料。

-------------------------

要了解如何使用 Go 编程与 GPT-3 进行交互的更多示例，请访问我们的 GitHub [仓库](https://github.com/Shubhamsaboo/kairos_gpt3/tree/master/Programming_with_GPT-3/GPT-3_Go)。

如何使用 Java 使用 OpenAI API？

Java 是用于开发传统软件系统的最古老和最流行的编程语言之一； 它还是一个带有运行时环境的平台。 它由 Sun Microsystems（现在是 Oracle 的子公司）在 1995 年开发，截至今日，超过 30 亿台设备运行在其上。 它是一种通用的、基于类的、面向对象的编程语言，旨在具有较少的实现依赖性。 其语法与 C 和 C++ 相似。 三分之二的软件行业仍然使用 Java 作为其核心编程语言。

让我们再次使用橄榄油文本摘要示例。 与我们在 Python 和 Go 中所做的一样，我们将向您展示如何调用 API，将训练提示作为请求发送，并使用 Java 将摘要完成作为输出。

要在本地计算机上逐步演示代码，请克隆我们的 GitHub [仓库](https://github.com/Shubhamsaboo/kairos_gpt3)。 在克隆的仓库中转到 Programming_with_GPT-3 文件夹，然后打开 GPT-3_Java 文件夹。

首先，导入所有相关的依赖项:

包示例;

// 导入依赖项

导入 java.util.*;

导入 java.io.*;

导入 com.theokanning.openai.OpenAiService;

导入 com.theokanning.openai.completion.CompletionRequest;

导入 com.theokanning.openai.engine.Engine;

现在，你将创建一个名为 OpenAiApiExample 的类。你所有的代码都将是它的一部分。在这个类下面，首先使用 API 令牌创建一个 OpenAiService 对象：

class OpenAiApiExample {

public static void main(String... args) throws FileNotFoundException {

String token = "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";

OpenAiService service = new OpenAiService(token);

现在已经建立了与 OpenAI API 的连接，形成了一个服务对象。从 prompts 文件夹中读取训练提示：

// 从 prompts 文件夹中读取训练提示

File file = new File("D:\\GPT-3 Book\\Programming with GPT-3\\GPT-3

Java\\example\\src\\main\\java\\example\\prompts\\summarize_for_a_2nd_grader.txt");

Scanner sc = new Scanner(file);

// 我们只需要使用 \\Z 作为分隔符

sc.useDelimiter("\\Z");

// pp 是由训练提示组成的字符串

String pp = sc.next();

接下来，您可以使用以下配置为 API 参数创建完成请求：

●        最大标记=100

●        执行引擎="Davinci"

●        温度=0.5

●        Top-p=1

●        频率惩罚=0.2

●        停止序列 = ["\n\n"]

// 创建一个字符串列表，用作停止序列

List<String> li = new ArrayList<String>();

li.add("\n\n'''");

// 创建一个包含 API 参数的完成请求

CompletionRequest completionRequest = CompletionRequest.builder().prompt(pp).maxTokens(100).temperature(0.5).topP(1.0).frequencyPenalty(0.2).stop(li).echo(true).build();

// 使用服务对象获取完成响应

service.createCompletion("davinci",completionRequest).getChoices().forEach(System.out::println);

将代码文件保存为 'text_summarization.java'，并在终端中运行它以生成输出。你可以使用以下命令从根目录运行文件：

(base) PS D:\GPT-3 Java> ./gradlew example:run

你的控制台应该打印出与之前示例相同的摘要。要了解如何使用 Java 编程与 GPT-3 交互的更多示例，请参阅我们的 GitHub [仓库](https://github.com/Shubhamsaboo/kairos_gpt3/tree/master/Programming_with_GPT-3/GPT-3_Java)。

GPT-3 Sandbox Powered by Streamlit

在本节中，我们将带领您了解 GPT-3 Sandbox，这是一个我们创建的开源工具，可以帮助您只需几行 Python 代码就能将您的想法变成现实。我们将向您展示如何使用它以及如何为您的特定应用程序进行定制。

我们的沙箱的目标是让您能够创建酷炫的 Web 应用程序，无论您的技术背景如何。它建立在 Streamlit 框架之上。

为了配合本书，我们还创建了一个 [视频系列](https://www.youtube.com/playlist?list=PLHdP3OXYnDmi1m3EQ76IrLoyJj3CHhC4M)，其中包含逐步说明如何创建和部署您的 GPT-3 应用的说明，您可以通过扫描第 3-1 图中的 QR 码来访问。请在阅读本章时参考它。

![](img/image-0-24.jpg)

图 3-1\. GPT-3 沙盒视频系列的 QR 码

我们在示例中使用 VSCode 作为 IDE，但请随意使用任何 IDE。在开始之前，你需要安装 IDE。请确保你正在运行 Python 版本 3.7 或更高版本。你可以通过运行以下命令确认你安装了哪个版本：

python --version

通过在你的 IDE 中打开新的终端并使用以下命令，从这个 [仓库](https://github.com/Shubhamsaboo/kairos_gpt3) 克隆代码：

git clone [`github.com/Shubhamsaboo/kairos_gpt3`](https://github.com/Shubhamsaboo/kairos_gpt3)

克隆仓库后，你的 IDE 中的代码结构现在应该如下所示：

![](img/image-0-25.jpg)

图 3-2\. 沙盒文件目录结构

你需要创建和部署 Web 应用所需的一切都已经在代码中了。你只需要调整一些文件以自定义沙盒以满足你特定的用例。

创建一个 [Python 虚拟环境](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)，你将其命名为 env。然后你可以安装所需的依赖项。

前往 email_generation 文件夹。你的路径应该像这样：

(env) kairos_gpt3\GPT-3 Sandbox\email_generation>

然后，运行以下命令：

(env) kairos_gpt3\GPT-3 Sandbox\email_generation> pip install -r requirements.txt

现在你可以开始定制沙盒代码了。你需要查看的第一个文件是 training_data.py。打开该文件并用你想要使用的训练提示替换默认提示。你可以使用 GPT-3 游乐场来尝试不同的训练提示（有关自定义沙盒的更多信息，请参见第二章以及我们的 [视频](https://www.youtube.com/watch?v=YGKY9Mc24MA&list=PLHdP3OXYnDmi1m3EQ76IrLoyJj3CHhC4M&index=3)）。

现在你可以根据你的应用用例的要求调整 API 参数（最大标记、执行引擎、温度、Top-p、频率惩罚、停止序列）。我们建议在游乐场中为给定的训练提示尝试不同的 API 参数值，以确定哪些值最适合你的用例。一旦你获得了满意的结果，然后你可以在 training_service.py 文件中更改值。

就这样！你的基于 GPT-3 的 Web 应用现在已经准备好了。你可以使用以下命令在本地运行它：

(env) kairos_gpt3\GPT-3 Sandbox\email_generation> streamlit run gpt_app.py

检查确保它能正常工作，然后你可以使用 Streamlit 分享将应用程序部署到互联网上，向更广泛的受众展示它。我们的 [视频](https://www.youtube.com/watch?v=IO2ndhOoTfc&list=PLHdP3OXYnDmi1m3EQ76IrLoyJj3CHhC4M&index=4) 提供了完整的部署步骤。

注意：本应用程序遵循简单的工作流程，其中训练提示从 UI 接收单个输入并生成响应。如果您的应用程序需要更复杂的工作流程，其中训练提示接收多个输入，请通过查看脚本 app1.py、app2.py 和 gpt_app.py 来自定义 UI 元素。有关详细信息，请参阅[Streamlit 文档](https://docs.streamlit.io)。

在接下来的几章中，我们将探索 GPT-3 的不同应用，并利用这个沙箱创建易于部署的 Web 应用程序。

结论

在本章中，我们学习了如何使用编程语言 Python、Go 和 Java 与 OpenAI API。我们还通过使用 Streamlit 创建的低代码沙箱环境进行了演示，这将帮助您快速将您的想法转化为应用程序。最后，我们看了一下上线 GPT-3 应用程序的关键要求。本章为您提供了 API 的编程视角；未来我们将深入探讨由 GPT-3 赋能的蓬勃发展的生态系统。
