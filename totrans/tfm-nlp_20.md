# 附录 IV — 使用 GPT-2 进行自定义文本完成

这个与*第七章*相关的附录，*GPT-3 引擎崛起的超人类变形车*，描述了如何使用 GPT-2 模型定制文本完成。

本附录展示了如何构建、训练 GPT-2 模型，并在 12 个步骤中与自定义文本进行交互。

打开本附录的 GitHub 仓库中的`Training_OpenAI_GPT_2.ipynb`。您会注意到笔记本也被分成了与本附录相同的 12 个步骤和单元格。

逐步运行笔记本中的每个单元格。这个过程是单调乏味的，但*克隆 OpenAI GPT-2 仓库产生的结果是令人满意的*。我们不会使用 GPT-3 API 或 Hugging Face 包装器。

我们会忙于了解模型是如何构建和训练的。您会看到一些弃用消息，但我们需要进入模型内部，而不是包装器或 API。然而，这个努力是值得的。

让我们开始激活 GPU。

# 训练 GPT-2 语言模型

在本节中，我们将在一个自定义数据集上训练一个 GPT-2 模型，然后与我们定制的模型进行交互。我们将使用与*第四章*中相同的`kant.txt`数据集，*从头开始预训练 RoBERTa 模型*。

我们将逐步打开笔记本并运行每个单元格。

## 步骤 1：先决条件

本节中提到的文件可以在本书的 GitHub 仓库的`AppendixIV`目录中找到：

+   如果您在 Google Colab 上运行它，请在 Google Colab 的笔记本运行时菜单中启用 GPU，就像*附录 III*中 *第 1 步：激活 GPU*中所解释的那样。

+   使用内置文件管理器将以下 Python 文件上传到 Google Colaboratory：`train.py`、`load_dataset.py`、`encode.py`、`accumulate.py`、`memory_saving_gradients.py`。

+   这些文件最初来自*N Shepperd*的 GitHub 仓库：[`github.com/nshepperd/gpt-2`](https://github.com/nshepperd/gpt-2)。但是，您可以从本书的 GitHub 仓库的`AppendixIV\``gpt-2-train_files`目录中下载这些文件。

+   *N Shepperd*的 GitHub 仓库提供了训练我们的 GPT-2 模型所需的文件。我们不会克隆*N Shepperd*的仓库，而是将从*N Shepperd*的仓库中获取的五个训练文件添加到 OpenAI 的仓库中。

+   用内置文件管理器将`dset.txt`上传到 Google Colaboratory。数据集被命名为`dset.txt`，这样在阅读本附录后，您可以用自定义输入替换其内容而无需修改程序。

+   这个数据集位于本附录的 GitHub 仓库中的`gpt-2-train_files`目录中。这是*第四章*中使用的`kant.txt`数据集，*从头开始预训练 RoBERTa 模型*。

我们现在将逐步进行训练过程的初始步骤。

## 第 2 至第 6 步：训练过程的初始步骤

本小节将仅简要介绍*步骤 2 到 6*，因为我们在*附录 III*，*使用 GPT-2 进行通用文本补全*中对其进行了详细描述。然后我们将把数据集和模型复制到项目目录中。

该程序现在克隆 OpenAI 的 GPT-2 仓库，而不是*N Shepperd*的仓库：

```py
#@title Step 2: Cloning the OpenAI GPT-2 Repository
#!git clone https://github.com/nshepperd/gpt-2.git
!git clone https://github.com/openai/gpt-2.git 
```

我们已经从*N Shepperd*的目录中上传了训练 GPT-2 模型所需的文件。

程序现在安装所需软件：

```py
#@title Step 3: Installing the requirements
import os             #when the VM restarts import os necessary
os.chdir("/content/gpt-2")    
!pip3 install -r requirements.txt 
```

该笔记本需要`toposort`，这是一种拓扑排序算法：

```py
!pip install toposort 
```

安装完所需软件后，请不要重新启动笔记本。相反，请等待直到您检查了 TensorFlow 版本，在您的会话期间只重新启动虚拟机一次。之后，如果有必要，则重新启动。深入了解代码而不仅仅是包装和 API 是繁琐但值得的。

现在我们检查 TensorFlow 版本，以确保我们正在运行`tf 1.x`版本：

```py
#@title Step 4: Checking TensorFlow version
#Colab has tf 1.x , and tf 2.x installed
#Restart runtime using 'Runtime' -> 'Restart runtime...'
%tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__) 
```

无论显示了`tf 1.x`版还是没有，请重新运行该单元格以确保，重新启动虚拟机，并重新运行该单元格。这样，您可以确保您正在运行带有`tf 1.x`的虚拟机。

该程序现在会下载我们将与我们的数据集训练的 117M 参数的 GPT-2 模型：

```py
#@title Step 5: Downloading 117M parameter GPT-2 Model
# run code and send argument
import os # after runtime is restarted
os.chdir("/content/gpt-2")
!python3 download_model.py '117M' #creates model directory 
```

我们将复制数据集和 117M 参数的 GPT-2 模型到`src`目录中：

```py
#@title Step 6: Copying the Project Resources to src
!cp /content/dset.txt /content/gpt-2/src/
!cp -r /content/gpt-2/models/ /content/gpt-2/src/ 
```

目标是将我们训练模型所需的所有资源分组到`src`项目目录中。

现在我们将浏览 N Shepperd 的训练文件。

## 第 7 步：N Shepperd 训练文件

我们将使用的训练文件来自*N Shepperd*的 GitHub 仓库。我们在本附录的*第 1 步：先决条件*中上传了它们。现在我们将把它们复制到我们的项目目录中：

```py
#@title Step 7: Copying the N Shepperd Training Files
#Referfence GitHub repository: https://github.com/nshepperd/gpt-2
import os # import after runtime is restarted
!cp /content/train.py /content/gpt-2/src/
!cp /content/load_dataset.py /content/gpt-2/src/
!cp /content/encode.py /content/gpt-2/src/
!cp /content/accumulate.py /content/gpt-2/src/
!cp /content/memory_saving_gradients.py /content/gpt-2/src/ 
```

现在训练文件已经准备好激活。让我们开始探索它们，首先是`encode.py`。

## 第 8 步：对数据集进行编码

在训练之前，数据集必须被编码。您可以双击`encode.py`在 Google Colaboratory 中查看文件。

`encode.py`通过调用`load_dataset.py`中的`load_dataset`函数加载`dset.txt`：

```py
from load_dataset import load_dataset
…/…
chunks = load_dataset(enc, args.in_text, args.combine, encoding=args.encoding) 
```

`encode.py`还加载 OpenAI 的编码程序`encode.py`来对数据集进行编码：

```py
import encoder
…/…
enc = encoder.get_encoder(args.model_name,models_dir) 
```

编码的数据集以`NumPy`数组的形式保存，并存储在`out.npz`中。现在，`npz`是由编码器生成的数组的`NumPy`压缩存档：

```py
import numpy as np
np.savez_compressed(args.out_npz, *chunks) 
```

当我们运行该单元格时，数据集将被加载、编码并保存在`out.npz`中：

```py
#@title Step 8:Encoding dataset
import os # import after runtime is restarted
os.chdir("/content/gpt-2/src/")
model_name="117M"
!python /content/gpt-2/src/encode.py dset.txt out.npz 
```

我们的 GPT-2 117M 模型已经准备好进行训练。

## 第 9 步：训练 GPT-2 模型

现在我们将对我们的数据集训练 GPT-2 117M 模型。我们将数据集的编码名称发送给程序：

```py
#@title Step 9:Training the Model
#Model saved after 1000 steps
import os # import after runtime is restarted
os.chdir("/content/gpt-2/src/")
!python train.py --dataset out.npz 
```

当您运行该单元格时，它将一直训练，直到您停止它。训练在 1,000 步后保存模型。当训练超过 1,000 步时，请停止它。保存的模型检查点位于`/content/gpt-2/src/checkpoint/run1`中。您可以在笔记本的*第 10A 步：复制训练文件*单元格中检查这些文件的列表。

您可以通过双击单元格的运行按钮来停止训练。训练将结束，并且训练参数将被保存。

您也可以在 1,000 步后停止训练模型，方法是使用*Ctrl* + *M*。程序将停止并保存训练参数。它会将代码转换为文本（您需要将其复制回代码单元格）并显示以下消息：

![](img/B17948_Appendix_IV_01.png)

图 IV.1：自动保存训练好的 GPT-2 模型

该程序使用`/content/gpt-2/src/memory_saving_gradients.py`和`/content/gpt-2/src/accumulate.py`程序来管理优化器和梯度。

`train.py`包含了可以调整以修改训练过程的完整参数列表。首先不要改变它们运行笔记本。然后，如果你愿意，可以尝试修改训练参数，看看能否获得更好的结果。

GPT-3 模型会在训练过程中生成一些样本供您阅读。在我训练 GPT-2 的过程中，系统生成了一个让我感到启迪的样本：

```py
The world is not a thing in itself, but is a representation of the world in itself. 
```

世界的形象是我们人类创造的，也是 AI 学到的。有趣！

让我们继续我们的实验，为我们的训练模型创建一个目录。

## 步骤 10：创建训练模型目录

本节将为我们的模型创建一个临时目录，存储我们需要的信息，并将其重命名以替换我们下载的 GPT-2 117M 模型目录。

我们首先创建一个名为`tgmodel`的临时目录：

```py
#@title Step 10: Creating a Training Model directory
#Creating a Training Model directory named 'tgmodel'
import os
run_dir = '/content/gpt-2/models/tgmodel'
if not os.path.exists(run_dir):
  os.makedirs(run_dir) 
```

然后我们复制包含我们训练模型时保存的训练参数的检查点文件，这是在本节的*步骤 9：训练模型*中进行的。

```py
#@title Step 10A: Copying training Files
!cp /content/gpt-2/src/checkpoint/run1/model-1000.data-00000-of-00001 /content/gpt-2/models/tgmodel
!cp /content/gpt-2/src/checkpoint/run1/checkpoint /content/gpt-2/models/tgmodel
!cp /content/gpt-2/src/checkpoint/run1/model-1000.index /content/gpt-2/models/tgmodel
!cp /content/gpt-2/src/checkpoint/run1/model-1000.meta /content/gpt-2/models/tgmodel 
```

我们的`tgmodel`目录现在包含我们的 GPT-2 模型的训练参数。

我们在*附录 III*的*使用 GPT-2 进行通用文本补全*的*步骤 5：下载 345M 参数 GPT-2 模型*中描述了这些文件的内容。

我们现在将从我们下载的 GPT-2 117M 模型中检索超参数和词汇文件：

```py
#@title Step 10B: Copying the OpenAI GPT-2 117M Model files
!cp /content/gpt-2/models/117M/encoder.json /content/gpt-2/models/tgmodel
!cp /content/gpt-2/models/117M/hparams.json /content/gpt-2/models/tgmodel
!cp /content/gpt-2/models/117M/vocab.bpe /content/gpt-2/models/tgmodel 
```

我们的`tgmodel`目录现在包含我们完整定制的 GPT-2 117M 模型。

我们的最后一步是将我们下载的原始 GPT-2 模型重命名，并将我们的模型名称设置为`117M`：

```py
#@title Step 10C: Renaming the model directories
import os
!mv /content/gpt-2/models/117M  /content/gpt-2/models/117M_OpenAI
!mv /content/gpt-2/models/tgmodel  /content/gpt-2/models/117M 
```

我们训练好的模型现在是克隆的 OpenAI GPT-2 代码库将要运行的模型。让我们与我们的模型交互吧！

## 步骤 11：生成无条件样本

在本节中，我们将与在我们的数据集上训练的 GPT-2 117M 模型进行互动。我们将首先生成一个无条件样本，不需要我们输入任何内容。然后，我们将输入一个上下文段落，以从我们训练好的模型获得一个条件文本补全响应。

让我们先运行一个无条件样本：

```py
#@title Step 11: Generating Unconditional Samples
import os # import after runtime is restarted
os.chdir("/content/gpt-2/src")
!python generate_unconditional_samples.py --model_name '117M' 
```

由于这是一个无条件的样本生成器，你不需要输入上下文句子。

要停止该单元格，双击单元格的运行按钮，或按*Ctrl* + *M*。

结果是随机的，但从语法的角度来看是合理的。从语义的角度来看，结果并不那么有趣，因为我们没有提供上下文。但仍然，这个过程是了不起的。它创造了帖子，写了一个标题，日期，想象了组织和地址，提出了一个主题，甚至想象了网页链接！

开头的几行令人难以置信：

```py
Title: total_authority
Category:
Style: Printable
Quote:
Joined: July 17th, 2013
Posts: 0
Offtopic link: "Essential research, research that supports papers being peer reviewed, research that backs up one's claims for design, research that unjustifiably accommodates scientific uncertainties, and research that persuades opens doors for science and participation in science",
href: https://groups.google.com/search?q=Author%3APj&src=ieKZP4CSg4GVWDSJtwQczgTWQhAWBO7+tKWn0jzz7o6rP4lEy&s sl=cTheory%20issue1&fastSource=posts&very=device
Offline
Joined: May 11th, 2014
Posts: 1729
Location: Montana AreaJoined: May 11th, 2014Posts: 1729Location: Montana
Posted: Fri Dec 26, 2017 9:18 pm Post subject: click
I. Synopsis of the established review group
The "A New Research Paradigm" and Preferred Alternative (BREPG) group lead authors John Obi (Australian, USA and Chartered Institute of Tropical and Climate Change Research), Marco Xiao (China and Department of Sociology/Ajax, International Institute of Tropical and Climate Change Research, Shanghai University) and Jackie Gu (US/Pacific University, Interselicitas de NASA and Frozen Planet Research Research Center, Oak Ridge National Laboratory). Dr. Obi states: "Our conclusions indicate that the existence of the new peer reviewed asan-rubie study predisposes journal publishers to read scientific publishers constantly to seek a consignment of, and to be affiliated with, a large target certain of their persons. The current practice of supplying books with peer review by commonly-overruled specialists itself spreads from part to part, sufficient to spread the impact of peer reviews by experts to all levels of the organization, so as to increase the extent of exposure to the term of deviation that source profiles or findings require". 
```

无条件的随机文本生成器的结果很有趣，但并不令人信服。

## 第 12 步：交互式上下文和补全示例

现在我们将运行一个有条件的示例。我们输入的上下文将使模型以我们想要的方式进行思考，通过生成定制段落来完成文本。

运行单元格并探索魔法：

```py
#@title Step 12: Interactive Context and Completion Examples
import os # import after runtime is restarted
os.chdir("/content/gpt-2/src")
!python interactive_conditional_samples.py --temperature 0.8 --top_k 40 --model_name '117M' 
```

如果需要，花几分钟回顾附录 III 的*第 9 步：与 GPT-2 交互*中的参数。程序提示我们输入上下文：

![](img/B17948_Appendix_IV_02.png)

图 IV.2：文本补全的上下文输入

让我们来看看埃曼纽尔·康德在附录 III 的*第 9 步：与 GPT-2 交互*中写的同一段落，*使用 GPT-2 进行通用文本补全*。

```py
Human reason, in one sphere of its cognition, is called upon to
consider questions, which it cannot decline, as they are presented by
its own nature, but which it cannot answer, as they transcend every
faculty of the mind. 
```

按下*Enter*键来生成文本，就像之前一样。尽管输出可能会从一个运行到另一个运行而变化，但它是有结构和逻辑的，这使得变压器具有吸引力。这次，结果不是随机的，而且令人印象深刻。

让我们来看看 GPT-2 模型生成的开头几行：

```py
"Now, what is the nature of the reason, except that it is of a kind and power of mind that is so called on to think, and to understand?
The reason is the thing which apprehends, and seeks to understand, the objects of its minds, which apprehend and apprehend the things of itself. No one can speak of this reason without the understanding and the understanding of the human reason. But that knowledge and understanding which is the most natural of all the faculties, which, because it is understood by the human reason, is the most natural of all the faculties, is not without reason, but only without understanding. The human reason, who sees and knows the things which the natural sense, which, though in the natural sense, does not understand, is the source of every other knowledge which we understand. The reason is the thing which does not understand, but understands the things which we comprehend, and therefore, is not to be understood. I mean that the reason is the thing which does not understand, but knows only the things which we do understand…." 
```

要停止单元格，双击单元格的运行按钮，或输入*Ctrl* + *M*。

哇！我怀疑有谁能看出我们训练过的 GPT-2 模型产生的文本补全与人类之间的区别。它也可能在每次运行时生成不同的输出。这可能是输出的样子，例如：

```py
The first question, to which it is obliged to respond, is, how can it be true that a mind which is ignorant of the truth of any proposition can, or ought, to, accept any proposition which it has not given it? And this question has been so well answered that it is impossible for any man to believe in the existence of any mind which has never received any information. 
"It is not the mind of any man, as far as we can judge, which is the subject of any philosophical inquiry. It is the mind of the minds of men, in their opinion, which they consider the most to be their most important subject. And if they can see through this, they will see it, and they will understand it, and they will understand it." 
```

我认为我们的模型在这个哲学、推理和逻辑的抽象练习中可能会胜过许多人类！

限制在于文本会随着每次运行而变化。因此，尽管看起来很出色，但它并不会满足我们在日常生活中的每一个需求。

我们可以从我们的实验中得出一些结论：

+   训练良好的变压器模型可以产生接近人类水平的文本补全。

+   在复杂和抽象推理方面，GPT-2 模型几乎可以达到人类水平的文本生成。

+   文本上下文是通过展示预期的内容来有效地对模型进行条件化的一种方法。

+   如果提供了上下文句子，文本补全就是基于文本条件生成文本。

+   尽管文本处于人类水平，但这并不意味着它会满足我们所有的需求。目前，它在局部是有趣的，但在整体上并不有效。

你可以尝试输入一些条件化的文本上下文示例来实验文本补全。你也可以用自己的数据训练我们的模型。只需用你自己的内容替换`dset.txt`文件的内容，然后看看会发生什么！

请记住，我们训练的 GPT-2 模型会像人类一样做出反应。如果你输入一个简短的、不完整的、不引人注意的或者棘手的上下文，你将会得到困惑或者糟糕的结果。这是因为 GPT-2 期望我们做到最好，就像在现实生活中一样！

# 参考资料

+   OpenAI GPT-2 GitHub 仓库: [`github.com/openai/gpt-2`](https://github.com/openai/gpt-2)

+   *N Shepperd* 的 GitHub 仓库: [`github.com/nshepperd/gpt-2`](https://github.com/nshepperd/gpt-2)

# 加入我们书籍的 Discord 空间

加入书籍的 Discord 工作空间，与作者进行每月的 *问我任何问题* 会议：

[`www.packt.link/Transformers`](https://www.packt.link/Transformers)

![](img/QR_Code5134042288713321484.png)
