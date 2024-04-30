# 9

# 匹配标记器和数据集

在研究变换器模型时，我们往往会关注模型的架构和提供给它们进行训练的数据集。我们已经探索了原始变换器，微调了一个类似 BERT 的模型，训练了一个 RoBERTa 模型，探索了一个 GPT-3 模型，训练了一个 GPT-2 模型，实现了一个 T5 模型等等。我们还研究了主要的基准任务和数据集。

我们训练了一个 RoBERTa 标记器并使用标记器对数据进行编码。然而，我们没有探索标记器的极限，以评估它们如何适应我们构建的模型。AI 是数据驱动的。*Raffel* 等人（2019），就像本书中引用的所有作者一样，花时间为变换器模型准备数据集。

在本章中，我们将介绍一些限制标记器的问题，这些问题妨碍了下游变换器任务的质量。不要轻易相信预训练的标记器。您可能有一个特定的词汇表（例如，高级医学术语），其中包含预训练标记器未处理的单词。

我们将从介绍一些与标记器无关的最佳实践开始，以衡量标记器的质量。我们将从标记化的角度描述数据集和标记器的基本准则。

然后，我们将使用 Word2Vec 标记器来查看标记器的限制，以描述我们在任何标记方法中面临的问题。这些限制将通过一个 Python 程序加以说明。

我们将继续通过在包含特定词汇的数据集上运行 GPT-2 模型来进行调查，包括无条件和有条件样本。

我们将进一步了解字节级 BPE 方法的限制。我们将构建一个显示由 GPT-2 标记器产生的结果的 Python 程序，并讨论在数据编码过程中出现的问题。这将表明，对于常见的 NLP 分析，并不总是需要 GPT-3 的优势。

然而，在本章末尾，我们将通过一个 **词性** (**POS**) 任务来测试一个 GPT-3 引擎，以查看模型理解的程度以及一个准备就绪的标记化字典是否符合我们的需求。

本章涵盖以下主题：

+   控制标记器输出的基本准则

+   原始数据策略和预处理数据策略

+   Word2Vec 标记化问题和限制

+   创建一个用于评估 Word2Vec 标记器的 Python 程序

+   构建一个用于评估字节级 BPE 算法输出的 Python 程序

+   使用特定词汇定制 NLP 任务

+   运行 GPT-2 的无条件和有条件样本

+   评估 GPT-2 标记器

我们的第一步将是探索 *Raffel* 等人（2019）定义的文本到文本方法论。

# 匹配数据集和标记器

下载基准数据集以训练变换器具有许多优点。数据已经准备好，并且每个研究实验室都使用相同的参考文献。此外，变换器模型的性能可以与具有相同数据的另一个模型进行比较。

然而，还需要做更多工作来改进 transformers 的性能。此外，在生产中实施 transformer 模型需要仔细规划和定义最佳实践。

在本节中，我们将定义一些最佳实践，以避免关键的障碍。

然后我们将通过在 Python 中使用余弦相似度来衡量分词和编码数据集的限制的几个示例。

让我们从最佳实践开始。

## 最佳实践

*Raffel*等人（2019）定义了一个标准的文本-文本 T5 transformer 模型。他们还走得更远。他们开始打破使用原始数据而不先进行预处理的神话。

预处理数据可以减少训练时间。例如，Common Crawl 包含通过网页提取获得的未标记文本。数据集中的非文本和标记已被删除。

然而，Google T5 团队发现，大部分通过 Common Crawl 获得的文本并不达到自然语言或英语的水平。因此他们决定在使用数据集之前需要对其进行清理。

我们将采纳*Raffel*等人（2019）提出的建议，并将企业质量控制最佳实践应用于预处理和质量控制阶段。在许多其他要应用的规则中，所描述的示例展示了为获得可接受的真实项目数据集所需的巨大工作。

*图 9.1*列出了应用于数据集的一些关键质量控制流程：

![](img/B17948_09_01.png)

图 9.1：transformer 数据集的最佳实践

如*图 9.1*所示，在训练 transformer 时，质量控制分为预处理阶段（*步骤 1*）和 transformer 投入生产后的质量控制（*步骤 2*）。

让我们浏览一下预处理阶段的主要方面。

### 步骤 1：预处理

*Raffel*等人（2019）建议在训练模型之前对数据集进行预处理，我加入了一些额外的想法。

Transformers 已经成为语言学习者，而我们已成为他们的老师。但是要教会一台机器学生一种语言，我们必须解释什么是正确的英语，例如。

在使用数据集之前，我们需要对其应用一些标准的启发式算法：

+   **句子令牌**

    建议选择以句号或问号结尾的句子。

+   **删除不良词汇**

    应该删除不良词汇。例如，可以在以下网站找到词汇列表：[`github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words`](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words)。

+   **删除代码**

    这有点棘手，因为有时代码就是我们正在寻找的内容。但是，通常最好从 NLP 任务的内容中删除代码。

+   **语言检测**

    有时，网站包含带有默认“lorem ipsum”文本的页面。有必要确保数据集的所有内容都是我们所希望的语言。一个很好的开始方法是使用 `langdetect`，它可以检测 50 多种语言: [`pypi.org/project/langdetect/`](https://pypi.org/project/langdetect/)。

+   **消除歧视引用**

    这是必须的。我的建议是建立一个知识库，其中包括您可以从网络上获取的所有内容或特定数据集。*压制任何形式的歧视*。您肯定希望您的机器是道德的!

+   **逻辑检查**

    将训练过的转换器模型应用于执行 **自然语言推理** (**NLI**) 的数据集可能是个好主意，以过滤掉毫无意义的句子。

+   **错误信息引用**

    消除指向无效链接、不道德网站或个人的文本。这是一项艰巨的工作，但肯定是值得的。

这个列表包含了一些主要的最佳实践。然而，还需要更多，比如过滤隐私法违规行为以及其他针对特定项目的行动。

一旦一个转换器被训练成学习正确的英语，我们需要帮助它在生产阶段检测输入文本中的问题。

### 步骤 2: 质量控制

一个训练过的模型将表现得像一个学习了语言的人一样。它将理解它可以理解的内容并从输入数据中学习。输入数据应该经过与 *Step 1: Preprocessing* 相同的过程，并将新信息添加到训练数据集中。反过来，训练数据集可以成为公司项目中的知识库。用户将能够在数据集上运行 NLP 任务并获得可靠的答案、特定文档的有用摘要等。

我们应该将 *Step 1: Preprocessing* 中描述的最佳实践应用到实时输入数据中。例如，一个转换器可以在来自用户或 NLP 任务的输入上运行，比如总结一系列文件。

转换器是有史以来最强大的 NLP 模型。这意味着我们的道德责任也增加了。

让我们来看一些最佳实践:

+   **实时检查输入文本**

    不要接受错误信息。而是实时解析输入并过滤不可接受的数据 (参见 *Step 1*)。

+   **实时消息**

    将被过滤的数据与被过滤原因一起存储，以便用户可以查阅日志。如果要求转换器回答不合适的问题，则显示实时消息。

+   **语言转换**

    当可能时，您可以将罕见的词汇转换为标准词汇。请参阅本章的 *Word2Vec 分词* 部分的 *Case 4*。这并不总是可能的。当可能时，它可能代表了一大步。

+   **隐私检查**

    无论您是将数据流入变压器模型还是分析用户输入，私人数据必须从数据集和任务中排除，除非经用户或所在国家授权。这是一个棘手的问题。必要时请咨询法律顾问。

我们刚刚浏览了一些最佳实践。现在让我们看看为什么人类质量控制是必要的。

### 连续的人类质量控制

变压器将逐渐接管大多数复杂的自然语言处理任务。然而，人类干预仍然是必需的。我们以为社交媒体巨头已经自动化了一切。然后我们发现有内容管理者决定了对他们平台上的内容的好坏。

正确的方法是训练一个变压器，实现它，控制输出，并将重要结果反馈到训练集中。因此，训练集将不断改进，变压器将继续学习。

*图 9.2* 显示了连续的质量控制如何帮助变压器的训练数据集增长并提高其在生产中的性能：

![](img/B17948_09_02.png)

图 9.2：连续的人类质量控制

我们已经浏览了 *Raffel* 等人（2019）描述的几种最佳实践，并根据我在企业人工智能项目管理方面的经验添加了一些指导。

让我们通过一个 Python 程序，举例说明一些分词器遇到的限制。

## Word2Vec 分词

只要一切顺利，没人会想到预训练分词器。就像在现实生活中一样。我们可以开车多年而不考虑引擎的问题。然后，有一天，我们的车抛锚了，我们试图找出原因来解释这种情况。

预训练分词器也是如此。有时结果并不如我们所期望的那样。例如，一些词对就是不匹配，正如我们在 *图 9.3* 中看到的：

![自动生成的图表说明](img/B17948_09_03.png)

图 9.3：分词器计算错误的单词对

*图 9.3* 中显示的例子来自 *美国独立宣言*、*权利法案* 和 *英国大宪章*：

+   `cake` 和 `chapters` 不匹配，尽管分词器将它们计算为具有较高余弦相似度值。

+   `freedom` 指的是言论自由，例如。 `copyright` 指的是免费电子书编辑的注释。

+   `pay` 和 `bill` 在日常英语中是匹配的。 `polysemy` 是指一个词可以有多个含义。例如，`Bill` 意味着要支付的金额，但也指的是 `权利法案`。结果是可以接受的，但这可能纯属运气。

在继续之前，让我们花点时间澄清一些问题。**QC**是指**质量控制**。在任何战略性企业项目中，质量控制都是强制性的。输出的质量将决定关键项目的生存。如果项目不是战略性的，错误有时是可以接受的。但在战略项目中，即使是少量的错误都意味着风险管理审计的介入，以确定项目是否应该继续还是放弃。

从质量控制和风险管理的角度来看，标记化不相关的数据集（太多无用词或缺少关键词）将混淆嵌入算法并产生“糟糕的结果”。这就是为什么在本章中，我将“标记化”一词使用宽泛，包括一些嵌入，因为前者对后者的影响。

在战略性的 AI 项目中，“糟糕的结果”可能是一个单一错误，但后果严重（特别是在医疗领域、飞机或火箭装配以及其他关键领域）。

打开`Tokenizer.ipynb`，基于我们在*第二章*、*开始使用 Transformer 模型架构*中创建的`positional_encoding.ipynb`。

由于 Word2Vec 算法的随机性质，结果可能因一次运行而异。

首先安装和导入了先决条件：

```py
#@title Pre-Requisistes
!pip install gensim==3.8.3
import nltk
nltk.download('punkt')
import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize 
import gensim 
from gensim.models import Word2Vec 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings(action = 'ignore') 
```

我们的数据集`text.txt`包含*美国独立宣言*、*权利法案*、*大宪章*、以及伊曼纽尔·康德的作品等其他文本。

现在将对`text.txt`进行标记化并训练一个 word2vec 模型：

```py
#@title Word2Vec Tokenization
#'text.txt' file
sample = open("text.txt", "r")
s = sample.read()
# processing escape characters
f = s.replace("\n", " ")
data = []
# sentence parsing
for i in sent_tokenize(f):
  temp = [] 
  # tokenize the sentence into words
  for j in word_tokenize(i):
    temp.append(j.lower())
  data.append(temp)
# Creating Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 512,window = 5, sg = 1)
print(model2) 
```

`window = 5`是一个有趣的参数。它限制了输入句子中当前单词和预测单词之间的*距离*。`sg = 1`表示使用了 skip-gram 训练算法。

输出显示词汇量的大小为`10816`，嵌入维度为`512`，学习速率设置为`alpha=0.025`：

```py
Word2Vec(vocab=10816, size=512, alpha=0.025) 
```

我们拥有一个带嵌入式的词汇表示模型，并且可以创建一个名为`similarity(word1,word2)`的余弦相似度函数。我们将`word1`和`word2`发送到该函数中，它会返回它们之间的余弦相似度值。值越高，相似度越高。

该函数首先会检测未知单词`[unk]`，并显示一条消息：

```py
#@title Cosine Similarity
def similarity(word1,word2):
        cosine=False #default value
        try:
                a=model2[word1]
                cosine=True
        except KeyError:     #The KeyError exception is raised
                print(word1, ":[unk] key not found in dictionary")#False implied
        try:
                b=model2[word2]#a=True implied
        except KeyError:       #The KeyError exception is raised
                cosine=False   #both a and b must be true
                print(word2, ":[unk] key not found in dictionary") 
```

只有在`cosine==True`时，才会计算余弦相似度，这意味着`word1`和`word2`都是已知的。

```py
 if(cosine==True):
                b=model2[word2]
                # compute cosine similarity
                dot = np.dot(a, b)
                norma = np.linalg.norm(a)
                normb = np.linalg.norm(b)
                cos = dot / (norma * normb)
                aa = a.reshape(1,512)
                ba = b.reshape(1,512)
                #print("Word1",aa)
                #print("Word2",ba)
                cos_lib = cosine_similarity(aa, ba)
                #print(cos_lib,"word similarity")

        if(cosine==False):cos_lib=0;
        return cos_lib 
```

该函数将返回`cos_lib`，余弦相似度的计算值。

我们现在将讨论六种情况。我们将把“数据集”命名为`text.txt`。

让我们从*案例 0*开始。

### 情况 0：数据集和词典中的单词

数据集中有`freedom`和`liberty`两个词，并可以计算它们的余弦相似度：

```py
#@title Case 0: Words in text and dictionary
word1="freedom";word2="liberty"
print("Similarity",similarity(word1,word2),word1,word2) 
```

相似性被限制为`0.79`，因为大量内容是从各种文本中插入的，以探索功能的限制：

```py
Similarity [[0.79085565]] freedom liberty 
```

相似度算法不是迭代确定性计算。这一部分的结果可能会因数据集内容、另一次运行后数据集大小或模块版本的变化而改变。如果你运行这个单元格 10 次，你可能会得到不同的值，就像以下的 10 次运行中一样。

在以下情况下，我使用 Google Colab VM 和 CPU 进行了 10 次实验，结果完全相同：

```py
Run 1: Similarity [[0.62018466]] freedom liberty
Run 2: Similarity [[0.62018466]] freedom liberty
...
Run 10: Similarity [[0.62018466]] freedom liberty 
```

然而，我在 Google Colab 的运行时菜单中做了一次“恢复出厂设置”。使用新的 VM 和 CPU，我得到了：

```py
Run 1: Similarity [[0.51549244]] freedom liberty
Run 2: Similarity [[0.51549244]] freedom liberty
...
Run 10: Similarity [[0.51549244]] freedom liberty 
```

我在 Google Colab 的运行时菜单中进行了另一次“恢复出厂设置”。我还激活了 GPU。使用新的 VM 和 GPU，我得到了：

```py
Run 1: Similarity [[0.58365834]] freedom liberty
Run 2: Similarity [[0.58365834]] freedom liberty
...
Run 10: Similarity [[0.58365834]] freedom liberty 
```

这里的结论是，随机算法基于概率。如果需要，运行预测`n`次是个好做法。

现在让我们看看当一个单词缺失时会发生什么。

### 情况 1：数据集或字典中没有的单词

缺少单词在许多方面都会带来麻烦。在这种情况下，我们将`corporations`和`rights`发送到相似度函数中：

```py
#@title Word(s) Case 1: Word not in text or dictionary
word1="corporations";word2="rights"
print("Similarity",similarity(word1,word2),word1,word2) 
```

字典中不包含单词`corporations`：

```py
corporations :[unk] key not found in dictionary
Similarity 0 corporations rights 
```

迷途！这个单词是一个未知的`[unk]`标记。

如果缺失的单词是重要的话，它将引发一系列事件和问题，扭曲变压器模型的输出。我们将这个缺失的单词称为`unk`。

需要检查几种可能性，并回答问题：

+   `unk`在数据集中，但没有被选中放入标记化字典中。

+   数据集中没有`unk`，这也适用于单词`corporations`。这解释了为什么在这种情况下它不在字典中。

+   如果用户发送一个包含该标记且未被标记化的输入给变压器，`unk`将会出现在生产中。

+   `unk`对于数据集来说不是一个重要的单词，但对变压器的使用是重要的。

如果变压器在某些情况下产生糟糕的结果，问题清单将继续增长。我们可以把`0.8`视为特定下游任务训练阶段变压器模型的出色性能。但在现实生活中，谁希望与一个错误率达到 20%的系统一起工作呢：

+   一个医生？

+   一个律师？

+   一个核电站维护团队？

`0.8` 在社交媒体等模糊环境中是令人满意的，因为很多消息本来就缺乏适当的语言结构。

现在是最糟糕的部分了。假设一个 NLP 团队发现了这个问题，并试图通过字节级 BPE 解决它，就像我们贯穿这本书所做的那样。如有必要，花几分钟回到*第四章*，*从头开始预训练 RoBERTa 模型*，*第三步：训练一个标记器*。

如果一个团队只使用字节级 BPE 来解决问题，噩梦就开始出现：

+   `unk` 将会被分解成单词片段。例如，我们可能得到`corporations`变成`corp` + `o` + `ra` + `tion` + `s`。其中一个或几个这样的单词片段在数据集中有很高的概率被发现。

+   `unk` 将变成一组由数据集中存在但不传达原始标记意义的标记表示的子词。

+   转换器将训练良好，没有人会注意到 `unk` 被分成片段并无意义地训练了。

+   转换器甚至可能会产生出色的结果，并将其性能从 `0.8` 提高到 `0.9`。

+   每个人都会鼓掌，直到专业用户在关键情况下应用了错误的结果。例如，在英语中，`corp` 可能是 `corporation` 或 `corporal`。这可能会导致 `corp` 与其他单词之间的混淆和不良关联。

我们可以看到，社交媒体的标准可能足以用于处理微不足道的主题的转换器。但是在现实生活中的企业项目中，将需要辛勤工作才能生成与数据集匹配的预训练标记器。在现实生活中，数据集每天都在随着用户输入而增长。用户输入成为应定期训练和更新的模型数据集的一部分。

例如，确保质量控制的一种方法可以通过以下步骤实现：

+   使用字节级 BPE 算法训练一个标记器。

+   使用类似于本章的“控制标记化数据”部分中将创建的程序来控制结果。

+   同样，训练一个使用于质量控制的 Word2Vec 算法的标记器，然后解析数据集，找到 `unk` 标记，并将其存储在数据库中。运行查询以检查是否缺少关键单词。

在如此详细地检查过程可能看起来是不必要的，并且你可能会倾向于依赖转换器对未见过的单词进行推理的能力。

然而，我建议在战略性项目中运行几种不同的质量控制方法，其中包括关键决策。例如，在法律摘要中，一个词可能是在法庭上赢得或输掉案件的区别。在航空航天项目（飞机，火箭）中，有一个 `0` 误差容忍标准。

运行越多的质量控制流程，你的转换器解决方案就会越可靠。

我们可以看到，获得可靠的数据集需要大量的工作！每一篇关于转换器的论文都以某种方式提到了制作可接受数据集所需的工作。

嘈杂的关系也会引起问题。

### 情况 2：嘈杂的关系

在这种情况下，数据集包含了单词 `etext` 和 `declaration`：

```py
#@title Case 2: Noisy Relationship
word1="etext";word2="declaration"
print("Similarity",similarity(word1,word2),word1,word2) 
```

此外，它们都出现在了标记化字典中：

```py
Similarity [[0.880751]] etext declaration 
```

更好的是，它们的余弦相似度似乎对其预测确信无疑，超过了 `0.5`。算法的随机性可能会导致在不同运行中产生不同的结果。

在微不足道的或社交媒体级别，一切看起来都很好。

然而，在专业水平上，结果是灾难性的！

`etext` 指的是 *古腾堡计划* 网站上每本电子书的前言，如本章的“匹配数据集和标记器”部分所解释的。特定任务的转换器目标是什么：

+   了解编辑的序言吗？

+   或者理解书的内容呢？

这取决于变压器的使用情况，可能需要几天的时间来解决。例如，假设编辑想要自动理解序言并使用变压器生成序言文本。我们应该将内容取出吗？

`declaration`是与*独立宣言*实际内容相关的有意义的词汇。

`etext`是*Project Gutenberg*添加到其所有电子书中的序言的一部分。

这可能会导致错误的自然语言推理，例如变压器被要求生成文本时产生*etext 是声明*。

让我们来看一个缺失单词的问题。

### 情况 3：文本中但不在字典中的词汇

在某些情况下，一个词可能在文本中但不在字典中。这将扭曲结果。

让我们来看看单词`pie`和`logic`：

```py
#@title Case 3: word in text, not in dictionary
word1="pie";word2="logic"
print("Similarity",similarity(word1,word2),word1,word2) 
```

单词`pie`不在字典中：

```py
pie :[unk] key not found in dictionary
Similarity 0 pie logic 
```

我们可以假设单词`pie`会在一个分词的字典中。但是如果没有或者另一个词没有呢？单词`pie`不在文本文件中。

因此，我们应该在流程中有函数来检测不在字典中的词汇，以实现更正或替代。此外，我们应该在流程中有函数来检测可能重要的数据集中的词汇。

让我们看看罕见词汇带来的问题。

### 情况 4：罕见词汇

罕见的词汇对超出简单应用范围的特定任务的变压器输出产生毁灭性影响。

管理罕见词汇延伸到许多自然语言的领域。例如：

+   罕见词汇可能出现在数据集中但被忽视，或者模型训练不足以处理它们。

+   罕见词汇可能是医学、法律、工程术语或任何其他专业行话。

+   罕见的词汇可能是俚语。

+   英语语言有数百种变体。例如，不同的英语词汇在美国、英国、新加坡、印度、澳大利亚和许多其他国家的某些地区使用。

+   罕见的词汇可能来自几个世纪前的文本，被遗忘或只有专家使用。

例如，在这种情况下，我们使用了单词`justiciar`：

```py
#@title Case 4: Rare words
word1="justiciar";word2="judgement"
print("Similarity",similarity(word1,word2),word1,word2) 
```

与`judgement`的相似性是合理的，但应该更高：

```py
Similarity [[0.6606605]] justiciar judgement 
```

你可能会认为单词`justiciar`有些牵强。分词器将其从*大宪章*中提取出来，可以追溯到 13 世纪初。不幸的是，程序会感到困惑，我们在每次运行后都会得到意外的结果。

注：预测可能会因为每次运行而有所不同。然而，它们显示了我们在变压器模型项目的分词和嵌入阶段中需要多么谨慎。

然而，*大宪章*的几条款在 21 世纪的英格兰仍然有效！例如，第 1、13、39 和 40 条仍然有效！

*大宪章*最著名的部分是以下摘录，它包含在数据集中：

```py
(39) No free man shall be seized or imprisoned, or stripped of his
rights or possessions, or outlawed or exiled, or deprived of his
standing in any other way, nor will we proceed with force against him,
or send others to do so, except by the lawful judgement of his equals
or by the law of the land.
(40) To no one will we sell, to no one deny or delay right or justice. 
```

如果我们在律师事务所中实施一个变压器模型来总结文件或其他任务，我们必须小心！

现在让我们看看解决稀有词问题的一些方法。

### Case 5: 替换稀有词

*替换稀有词本身就是一个项目*。这项工作保留给特定的任务和项目。假设企业预算可以支付航空领域的知识库成本，例如。在这种情况下，值得花费必要的时间来查询分词目录，以找到它错过的单词。

问题可以按主题分组解决，并且知识库将定期更新。

在*Case 4*中，我们遇到了单词`justiciar`。如果我们回到它的起源，我们可以看到它来自法国诺曼德语，并且是法国类拉丁语单词`judicaire`的根源。

我们可以用`judge`替换`justiciar`，这传达了相同的元概念：

```py
#@title Case 5: Replacing rare words
word1="judge";word2="judgement"
print("Similarity",similarity(word1,word2),word1,word2) 
```

它产生了一个有趣的结果，但由于算法的非确定性方面，我们仍然需要小心：

```py
Similarity [[0.7962761]] judge judgement 
```

我们也可以保留单词`justiciar`，但尝试单词的现代含义并将其与`judge`进行比较。您可以尝试实现`Case 5: Replacing rare words`：

```py
word1="justiciar";word2="judge"
print("Similarity",similarity(word1,word2),word1,word2) 
```

无论如何，一些稀有词都需要被更常见的词替换。

结果将是令人满意的：

```py
Similarity [[0.9659128]] justiciar judge 
```

我们可以创建使用替换单词的查询，直到我们找到相关性超过`0.9`的情况。此外，如果我们正在管理一个关键的法律项目，我们可以将包含任何类型稀有词的重要文档翻译成标准英语。因此，变换器在自然语言处理任务中的性能将提高，并且公司的知识库将逐渐增加。

现在让我们看看如何使用余弦相似度进行蕴含验证。

### Case 6: 蕴涵

在这种情况下，我们对字典中的单词感兴趣，并按固定顺序测试它们。

例如，让我们看看“`pay`" + “`debt`"是否在我们的相似性函数中有意义：

```py
#@title Case 6: Entailment
word1="pay";word2="debt"
print("Similarity",similarity(word1,word2),word1,word2) 
```

结果令人满意：

```py
Similarity [[0.89891946]] pay debt 
```

我们可以检查数据集中的几对单词，并检查它们是否有意义。例如，这些单词对可以从法律部门的电子邮件中提取。如果余弦相似度超过`0.9`，则可以剥离邮件中的无用信息，并将内容添加到公司的知识库数据集中。

现在让我们看看预训练的分词器与自然语言处理任务的匹配情况。

# 具有特定词汇的标准自然语言处理任务

本节重点介绍了本章节中*Word2Vec tokenization*部分的*Case 4: Rare words*和*Case 5: Replacing rare words*。

我们将使用`Training_OpenAI_GPT_2_CH09.ipynb`，这是我们在*Chapter 7*中使用的笔记本的重命名版本，*The Rise of Suprahuman Transformers with GPT-3 Engines*。

笔记本有两处更改：

+   `dset`，数据集，被重命名为`mdset`，并包含医学内容

+   添加了一个 Python 函数来控制使用字节级 BPE 分词的文本

我们不会描述*第七章*中所涵盖的`Training_OpenAI_GPT_2_CH09.ipynb`，以及*附录 III 和 IV*。确保在开始之前上传所需的文件，就像*第七章*中所解释的那样。

您希望训练模型的时间没有限制。中断它以保存模型。

文件位于`Chapter09`的`gpt-2-train_files`目录中的 GitHub 上。尽管我们使用的是*第七章*中相同的笔记本，但要注意数据集`dset`在目录和代码中现在被命名为`mdset`。

首先，让我们使用经过训练以理解医学内容的 GPT-2 模型生成一个无条件样本。

## 使用 GPT-2 生成无条件样本

在这一节中，我们将亲自动手来理解变换器的内部工作原理。当然，我们可以跳过整个章节，简单地使用 OpenAI API。然而，一个 4.0 的 AI 专家必须成为 AI 大师，通过预处理管道*展示*，而不是模糊地告诉变换器模型要做什么。为了*展示*一个变换器模型要做什么，必须了解变换器模型的工作原理。

在*案例 4：罕见词语*和*案例 5：替换罕见词语*中，我们看到罕见词语可以是在特定领域中使用的词语，古老的英语，世界各地英语的变体，俚语等。

在 2020 年，新闻中充斥着与 COVID-19 爆发有关的医学术语。在这一部分中，我们将看到一个 GPT-2 变换器如何处理医学文本。

要编码和训练的数据集包含了*Martina Conte*和*Nadia Loy*（2020 年）撰写的一篇论文，名称为*具有非局部感知的多线索动力学模型用于细胞在具有趋化作用的纤维网络上的迁移*。

标题本身并不容易理解，包含了一些罕见的词语。

加载位于`gpt-2-train_files`目录中的文件，包括`mdset.txt`。然后按照*第七章*中所述运行代码。您可以逐个单元格地运行此代码，*第七章*会给出指导。务必按照说明确保激活`tf 1.x`。在运行*第 4 步*之后，务必重新启动运行时，然后再次运行*第 4 步*中的`tf 1.x`单元格，然后再继续。否则，您将在笔记本中遇到错误。在本节中，我们将亲自动手使用低级别的原始 GPT-2 代码，而不是 API。

在对医学数据集进行训练之后，您将进入无条件抽样单元，*第 11 步：生成无条件样本*：

```py
#@title Step 11: Generating Unconditional Samples
import os # import after runtime is restarted
os.chdir("/content/gpt-2/src")
!python generate_unconditional_samples.py --model_name '117M' 
```

此命令以及本笔记本中的其他代码运行所需的时间取决于您的计算机性能。本书中的所有 GPT-2 代码都仅用于教育目的。建议在生产环境中使用 OpenAI 的 GPT-3 API。对于变换器项目，响应时间更快。

运行单元格，并在需要时停止。它会产生一个随机输出：

```py
community-based machinery facilitates biofilm growth. Community members place biochemistry as the main discovery tool to how the cell interacts with the environment and thus with themselves, while identifying and understanding all components for effective Mimicry.
2\. Ol Perception
Cytic double-truncation in phase changing (IP) polymerases (sometimes called "tcrecs") represents a characteristic pattern of double-crossing enzymes that alter the fundamental configuration that allows initiation and maintenance of process while chopping the plainNA with vibrational operator. Soon after radical modification that occurred during translational parasubstitution (TMT) achieved a more or less uncontrolled activation of SYX. TRSI mutations introduced autophosphorylation of TCMase sps being the most important one that was incorporated into cellular double-triad (DTT) signaling across all
cells, by which we allow R h and ofcourse an IC 2A- >
.../... 
```

仔细观察输出，我们注意到以下几点：

+   生成的句子的结构相对可接受

+   输出的语法不错

+   对于非专业人士来说，输出可能看起来类似于人类

但是内容毫无意义。变压器无法产生与我们训练的医学论文相关的真实内容。要获得更好的结果需要艰苦的工作。当然，我们总是可以增加数据集的规模。但它是否包含我们正在寻找的内容呢？我们是否会找到更多数据中的错误相关性呢？例如，想象一下一个涉及 COVID-19 的医疗项目，数据集包含以下句子：

+   `COVID-19 不是危险的病毒，而只是像普通流感一样`。

+   `COVID-19 是一种非常危险的病毒`。

+   `COVID-19 不是一种病毒，而是实验室创造出的东西`。

+   `COVID-19 肯定不是由实验室创造的!`

+   `疫苗是危险的!`

+   `疫苗是救命稻草!`

+   `政府没有正确管理疫情`。

+   `政府采取了必要的措施`。

以及更多类似这样矛盾的句子。这些不一致性证实了数据集和分词器都必须为专业的医疗保健项目、航空航天、交通运输和其他关键领域进行定制。

想象一下，你有数十亿字的数据集，但内容如此矛盾和嘈杂，无论你如何尝试，都无法得到可靠的结果！

这可能意味着数据集必须更小，限制在科学论文的内容上。但即便如此，科学家们对彼此之间也常常意见不一。

结论是，要产生可靠的结果需要大量的辛勤工作和一支牢固的团队。

现在让我们尝试对 GPT-2 模型进行条件化设置。

## 生成经过训练的条件样本

在本节中，我们转到笔记本的*步骤 12：交互式上下文和完成示例*单元格，并运行它：

```py
#@title Step 12: Interactive Context and Completion Examples
import os # import after runtime is restarted
os.chdir("/content/gpt-2/src")
!python interactive_conditional_samples.py --temperature 0.8 --top_k 40 --model_name '117M' --length 50 
```

工业 4.0 人工智能专家将更少地关注代码，更多地关注如何*展示*变压器模型做什么。每个模型都需要一定程度的指导，而不仅仅是使用无条件的数据来模糊地告诉它做某事。

我们通过输入医学论文的一部分来对 GPT-2 模型进行条件设定：

```py
During such processes, cells sense the environment and respond to external factors that induce a certain direction of motion towards specific targets (taxis): this results in a persistent migration in a certain preferential direction. The guidance cues leading to directed migration may be biochemical or biophysical. Biochemical cues can be, for example, soluble factors or growth factors that give rise to chemotaxis, which involves a mono-directional stimulus. Other cues generating mono-directional stimuli include, for instance, bound ligands to the substratum that induce haptotaxis, durotaxis, that involves migration towards regions with an increasing stiffness of the ECM, electrotaxis, also known as galvanotaxis, that prescribes a directed motion guided by an electric field or current, or phototaxis, referring to the movement oriented by a stimulus of light [34]. Important biophysical cues are some of the properties of the extracellular matrix (ECM), first among all the alignment of collagen fibers and its stiffness. In particular, the fiber alignment is shown to stimulate contact guidance [22, 21]. TL;DR: 
```

我们在输入文本的结尾加上`TL;DR`：告诉 GPT-2 模型总结我们对它进行条件化的文本。输出在语法和语义上都是有意义的：

```py
the ECM of a single tissue is the ECM that is the most effective.
To address this concern, we developed a novel imaging and immunostaining scheme that, when activated, induces the conversion of a protein to its exogenous target 
```

由于输出是非确定性的，我们也可能得到这样的回答：

```py
Do not allow the movement to be directed by a laser (i.e. a laser that only takes one pulse at a time), but rather a laser that is directed at a target and directed at a given direction. In a nutshell, be mindful. 
```

结果更好，但需要更多的研究。

从这个例子和章节中我们可以得出的结论是，对于预训练的变压器模型，例如在大量随机网络爬行数据上进行预训练，将教导变压器模型英语。然而，就像我们一样，变压器也需要在特定领域接受训练，才能成为该领域的专家。

让我们进一步调查并控制分词化的数据。

## 控制分词化的数据

本节将读取 GPT-2 模型使用其预训练分词器编码的前面词语。

运行单元格时，在运行后续单元格之前停止一个单元格。

我们将转到本章中使用的 `Training_OpenAI_GPT_2_CH09.ipynb` 笔记本的 `Additional Tools: Controlling Tokenized Data` 单元格。该单元格是为本章添加到笔记本中的。

该单元首先解压 `out.npz`，其中包含编码的医学论文，该论文位于数据集 `mdset` 中：

```py
#@title Additional Tools : Controlling Tokenized Data
#Unzip out.npz
import zipfile
with zipfile.ZipFile('/content/gpt-2/src/out.npz', 'r') as zip_ref:
    zip_ref.extractall('/content/gpt-2/src/') 
```

解压 `out.npz`，我们可以读取 `arr_0.npy`，包含我们正在寻找的编码数据集的 `NumPy` 数组：

```py
#Load arr_0.npy which contains encoded dset
import numpy as np
f=np.load('/content/gpt-2/src/arr_0.npy')
print(f)
print(f.shape)
for i in range(0,10):
    print(f[i]) 
```

输出是数组的前几个元素：

```py
[1212 5644  326 ...   13  198 2682] 
```

现在我们将打开 `encoder.json` 并将其转换为 Python 字典：

```py
#We first import encoder.json
import json
i=0
with open("/content/gpt-2/models/117M/encoder.json", "r") as read_file:
    print("Converting the JSON encoded data into a Python dictionary")
    developer = json.load(read_file) #converts the encoded data into a Python dictionary
    for key, value in developer.items(): #we parse the decoded json data
        i+=1
        if(i>10):
            break;
        print(key, ":", value) 
```

最后，我们显示了我们编码数据集的前 `500` 个标记的键和值：

```py
#We will now search for the key and value for each encoded token
    for i in range(0,500):
        for key, value in developer.items():
            if f[i]==value:
                print(key, ":", value) 
```

`mdset.txt` 的前几个单词如下：

```py
This suggests that 
```

我添加了这些单词以确保 GPT-2 预训练的分词器能够轻松识别它们，这也确实是这样的：

```py
This : 1212
Ġsuggests : 5644
Ġthat : 326 
```

我们可以轻松识别前导空格字符（`Ġ`）前的初始标记。然而，让我们看一下医学论文中的下一个词：

```py
amoeboid 
```

`amoeboid` 是一个罕见的词。我们可以看到 GPT-2 的分词器将其分解为子词：

```py
Ġam : 716
o : 78
eb : 1765
oid : 1868 
```

让我们跳过空格，看看发生了什么。`amoeboid` 变成了 `am` + `o`+ `eb` + `oid`。我们必须同意，没有未知的标记：`[unk]`。这是由于使用了字节级 BPE 策略。

然而，变压器的注意力层可能会关联：

+   `am` 与其他序列，如 `I am`

+   `o` 与任何包含 `o` 的序列

+   `oid` 与另一个包含 `oid` 的序列，可能与某些算法的 `tabloid` 相关

这一点一点都不好。让我们进一步看看以下单词：

```py
amoeboid and mesenchymal 
```

输出清晰地显示了 `and`。至于其余的部分，标记令人困惑：

```py
Ġam : 716
o : 78
eb : 1765
oid : 1868
Ġand : 290
Ġmes : 18842
ench : 24421
ym : 4948
al : 282 
```

你可能想知道为什么这是个问题。原因可以用一个词来概括：多义性。如果我们使用 word2vec 分词器，词典可能不包含罕见的词语，比如 `amoeboid`，我们将得到一个未知的标记。

如果我们使用字节级 BPE，我们会得到更好的结果，因为我们排除了更少的同一单词的变体，比如 `go` 和 `go` + `ing`。

然而，`amoeboid` 中的 `am` 标记在低级别带来了多义性的问题。`am` 可以是一种前缀，像 `I` + `am` 中的 `am`，或者像 `am` + `bush` 中的子词。注意层可能会将一个标记的 `am` 关联到另一个 `am`，从而创建不存在的关系。这定义了自然语言理解中多义性的核心问题。

我们可以说进展正在取得，但我们需要更努力地改进自然语言处理。

我们已经通过一些例子了解了我们在实际项目中面临的很多日常问题。花些时间尝试一些你认为有用的例子。

在我们离开之前，我们将使用一个探测任务来验证变压器模型提供的自然语言理解水平。

# 探索 GPT-3 的范围

即使是像 OpenAI GPT-3 这样最强大的转换器也有它们的局限性。让我们看看 GPT-3 如何对待 `amoeboid` 这个词，它更接近于医学术语而不是主流词汇。在许多项目中，我们需要技术术语。匹配数据集需要质量控制来确保转换器如何组织其字典和嵌入。

我们人类可以检测错误并纠正别人。例如，在本章的 *控制标记化数据* 部分中，我们探讨了 `amoeboid` 这个词。

让我们首先问问 GPT-3 `amoeboid` 是什么意思：

![图形用户界面、文本、应用程序、电子邮件 自动生成的描述](img/B17948_09_04.png)

图 9.4：询问 GPT-3 “amoeboid” 的含义

`amoeboid`（类似变形虫）是一个形容词，但 GPT-3 在输出中表示它是一个名词：

```py
A: Amoeboid is a noun which means "resembling an amoeba" 
```

然后我们向 GPT-3 提出一个更加精确的问题，但仍然得到一个错误的答案：

```py
Q: Is amoeboid a noun or an adjective?
A: Amoeboid is a noun. 
```

最后，我们坚持要求清晰的定义，并得到了正确的答案：

```py
Q: What does amoeboid mean in medical terms? 
A: Amoeboid means "resembling an amoeba". 
```

定义是准确的，尽管语法分析不准确。

在现实项目中，什么更重要？是理解一个词的定义，还是确定它在句子中作为形容词还是名词的角色？

一个词的定义对于医学项目来说已经足够了。在这种情况下，GPT-3 可能已经足够了。如果定义已经足够，那么语义角色标注不是理解句子的先决条件。

也许语法方面对于教育语法学校项目很重要，但对于企业供应链、金融和电子商务应用程序来说并不重要。

OpenAI GPT-3 在两种情况下都可以进行微调，正如我们在 *第七章*，*GPT-3 引擎的超人类转变* 中所看到的那样。

本节总结认为，我们必须确保在训练好的转换器模型中拥有所有需要的数据。如果没有，标记化过程将是不完整的。也许我们应该拿出一本医学词典，并创建一个包含特定词汇的大型医学文章语料库。然后，如果模型仍然不够准确，我们可能需要对数据集进行标记化并从头开始训练模型。

一个 2022 年的开发者将会有较少的开发工作，但仍然需要大量思考和设计！

现在让我们结束本章，转向另一个 NLU 任务。

# 总结

在本章中，我们衡量了标记化和后续数据编码过程对转换器模型的影响。一个转换器模型只能关注堆栈中的嵌入和位置编码子层中的标记。这个模型是编码器-解码器、仅编码器还是仅解码器模型并不重要。数据集看起来足够好训练也不重要。

如果标记化过程失败，即使只是部分失败，我们正在运行的转换器模型也会错过关键的标记。

我们首先发现，对于标准语言任务来说，原始数据集可能足以训练一个转换器。

但是，我们发现，即使预训练的标记化器经过了十亿字的训练，它只会创建一个很小的词汇表，其中包含了它遇到的词汇的一小部分。像我们一样，标记化器捕捉到了它正在学习的语言的本质，只有这些词汇也经常被使用，它才会*记住*最重要的词汇。这种方法对于标准任务效果很好，但在特定任务和词汇方面会出现问题。

然后，我们寻找了一些想法，其中之一是解决标准标记化器的限制。我们应用了一种语言检查方法，以适应我们希望处理的文本，比如一个标记化器*思考*和编码数据的方式。

我们将该方法应用于 GPT-2 的无条件和有条件任务。

最后，我们分析了数据标记化和匹配数据集与 GPT-3 的限制。从这一章可以得到的教训是，AI 专家将在相当长的一段时间内存在！

在下一章中，*基于 BERT 的变压器的语义角色标注*，我们将深入探讨 NLU，并使用 BERT 模型要求变压器解释句子的含义。

# 问题

1.  一个标记化的字典包含了语言中存在的每一个单词。（真/假）

1.  预训练的标记化器可以对任何数据集进行编码。（真/假）

1.  在使用数据库之前检查数据库是一个好的做法。（真/假）

1.  从数据集中清除淫秽数据是一个好的做法。（真/假）

1.  删除包含歧视性言论的数据是一个好的做法。（真/假）

1.  原始数据集有时可能产生嘈杂内容和有用内容之间的关系。（真/假）

1.  一个标准的预训练标记化器包含了过去 700 年的英语词汇。（真/假）

1.  老式英语可能在用现代英语训练的标记化器对数据进行编码时产生问题。（真/假）

1.  医学和其他类型的行话可能在用现代英语训练的标记化器对数据进行编码时产生问题。（真/假）

1.  控制预训练标记化器产生的编码数据的输出是一个好的做法。（真/假）

# 参考资料

+   *Colin Raffel*、*Noam Shazeer*、*Adam Roberts*、*Katherine Lee*、*Sharan Narang*、*Michael Matena*、*Yanqi Zhou*、*Wei Li* 和 *Peter J. Liu*，2019 年，《探索具有统一文本到文本转换器的迁移学习的极限》：[`arxiv.org/pdf/1910.10683.pdf`](https://arxiv.org/pdf/1910.10683.pdf)

+   OpenAI GPT-2 GitHub 代码库：[`github.com/openai/gpt-2`](https://github.com/openai/gpt-2)

+   N. Shepperd 的 GitHub 代码库：[`github.com/nshepperd/gpt-2`](https://github.com/nshepperd/gpt-2)

+   Hugging Face 框架和资源：[`huggingface.co/`](https://huggingface.co/)

+   美国法律，*蒙大拿州企业法*：[`corporations.uslegal.com/state-corporation-law/montana-corporation-law/#:~:text=Montana%20Corporation%20Law,carrying%20out%20its%20business%20activities`](https://corporations.uslegal.com/state-corporation-law/montana-corporation-law/#:~:text=Montana%20Corporation%20Law,carrying%20out%20its%20business%20activities)

+   *玛蒂娜·孔特*，*娜迪亚·洛伊*，2020 年，《具有非局部感知的多线索动力学模型用于化学趋化作用纤维网络上的细胞迁移》：[`arxiv.org/abs/2006.09707`](https://arxiv.org/abs/2006.09707)

+   *美利坚合众国独立宣言*，由*托马斯·杰斐逊*：[`www.gutenberg.org/ebooks/1`](https://www.gutenberg.org/ebooks/1)

+   *美国权利法案*，由美国及相关文本：[`www.gutenberg.org/ebooks/2`](https://www.gutenberg.org/ebooks/2)

+   *大宪章*：[`www.gutenberg.org/ebooks/10000`](https://www.gutenberg.org/ebooks/10000)

+   *《纯粹理性批判》*，*《实践理性批判》*和*《道德形而上学基本原理》*：[`www.gutenberg.org`](https://www.gutenberg.org)

# 加入我们书籍的 Discord 空间。

加入该书的 Discord 工作区，与作者进行每月的*问我任何*活动。

[`www.packt.link/Transformers`](https://www.packt.link/Transformers)

![](img/QR_Code5134042288713321484.png)
