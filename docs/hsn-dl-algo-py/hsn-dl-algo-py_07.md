# 第五章：RNN 的改进

**递归神经网络**（**RNN**）的缺点是它不能长时间保持信息在内存中。我们知道 RNN 将信息序列存储在其隐藏状态中，但是当输入序列过长时，由于消失梯度问题，它无法将所有信息保留在内存中，这是我们在前一章节讨论过的。

为了应对这个问题，我们引入了一种称为**长短期记忆**（**LSTM**）单元的 RNN 变体，通过使用称为**门**的特殊结构来解决消失梯度问题。门会根据需要保持信息在内存中。它们学会了什么信息应该保留，什么信息应该丢弃。

我们将从探索 LSTM 和它如何克服 RNN 的缺点开始这一章节。接下来，我们将学习如何使用 LSTM 单元在 TensorFlow 中执行前向和后向传播，并且了解如何使用它们来预测比特币的价格。

接下来，我们将掌握**门控循环单元**（**GRU**）单元，它是 LSTM 单元的简化版本。我们将学习 GRU 单元中的前向和后向传播的工作原理。接下来，我们将基本了解双向 RNNs 的工作原理及其如何利用过去和未来的信息进行预测；我们还将了解深层 RNNs 的工作方式。

在章节结束时，我们将学习关于序列到序列模型的内容，该模型将长度可变的输入映射到长度可变的输出。我们将深入探讨序列到序列模型的架构和注意力机制。

为了确保对这些主题有清晰的理解，本章节按以下方式组织：

+   LSTM 来拯救

+   LSTM 中的前向和后向传播

+   使用 LSTM 预测比特币价格

+   GRUs

+   GRUs 中的前向和后向传播

+   双向 RNNs

+   深层 RNNs

+   序列到序列模型

# LSTM 来拯救

在反向传播 RNN 时，我们发现了一个称为**消失梯度**的问题。由于消失梯度问题，我们无法正确训练网络，导致 RNN 无法在记忆中保留长序列。为了更好地理解我们所说的，让我们考虑一个简短的句子：

*天空是 __*。

基于它所见的信息，RNN 可以轻松预测空白为 *蓝色*，但它无法涵盖长期依赖关系。这意味着什么？让我们考虑以下句子以更好地理解这个问题：

*阿奇在中国生活了 13 年。他喜欢听好音乐。他是漫画迷。他精通 ____。*

现在，如果我们被要求预测前述句子中的缺失词，我们会预测它为*Chinese*，但是我们是如何预测的呢？我们简单地记住了前面的句子，并理解阿奇在中国生活了 13 年。这使我们得出结论，阿奇可能会说流利的中文。另一方面，RNN 无法在其记忆中保留所有这些信息以表明阿奇能说流利的中文。由于梯度消失问题，它无法长时间保留信息。也就是说，当输入序列很长时，RNN 的记忆（隐藏状态）无法容纳所有信息。为了缓解这一问题，我们使用 LSTM 单元。

LSTM 是 RNN 的一种变体，解决了梯度消失问题，并在需要时保留信息在内存中。基本上，图中的隐藏单元中的 RNN 单元被替换为 LSTM 单元：

![](img/83fc7036-f6b4-47d1-bc44-02569007c4fc.png)

# 理解 LSTM 单元

是什么使 LSTM 单元如此特殊？LSTM 单元如何实现长期依赖？它如何知道什么信息需要保留，什么信息需要从记忆中丢弃？

这一切都是通过称为**门**的特殊结构实现的。如下图所示，典型的 LSTM 单元包括三个特殊的门，称为输入门、输出门和遗忘门：

![](img/5bed01df-ee73-4281-9a32-73366390c448.png)

这三个门负责决定什么信息需要添加、输出和从记忆中遗忘。有了这些门，LSTM 单元可以有效地只在需要时保留信息在记忆中。

在 RNN 单元中，我们使用隐藏状态，![](img/a670126d-efa7-49a6-9b47-72b41e459bc0.png)，有两个目的：一个是存储信息，另一个是进行预测。不像 RNN，在 LSTM 单元中，我们将隐藏状态分解为两个状态，称为**单元状态**和**隐藏状态**：

+   单元状态也称为内部存储器，所有信息将存储在这里

+   隐藏状态用于计算输出，也就是进行预测。

单元状态和隐藏状态在每个时间步共享。现在，我们将深入研究 LSTM 单元，看看这些门如何使用以及隐藏状态如何预测输出。

# 遗忘门

遗忘门，![](img/e6ab9a8d-a968-4f98-bf96-8996e768b599.png)，负责决定应该从单元状态（记忆）中移除哪些信息。考虑以下句子：

*哈里* *是一位好歌手*。*他住在纽约。赛恩也是一位好歌手*。

一旦我们开始谈论赛恩，网络就会理解主题已从哈里变为赛恩，关于哈里的信息不再需要。现在，遗忘门将从单元状态中删除/遗忘有关哈里的信息。

遗忘门由 sigmoid 函数控制。在时间步![](img/5eb5937b-5a0a-4fa5-a9f4-a54ee188382a.png)，我们将输入![](img/4b86f429-7089-4dfb-a350-f87a071613d4.png)和上一个隐藏状态![](img/518639c9-dc9e-4590-8132-afb793ff6307.png)传递给遗忘门。如果细胞状态中的特定信息应该被移除，则返回`0`，如果不应该被移除，则返回`1`。在时间步![](img/b0d40b0d-15e7-41ea-bef2-59f3ca46dcec.png)，遗忘门![](img/b008e595-73f5-4779-9bb9-f0c13f921f38.png)的表达如下：

![](img/e666c2a3-5145-44aa-9f12-9cc4e579e2f8.png)

在这里，适用以下内容：

+   ![](img/b50ec728-ddb0-4569-bc15-862460d757d3.png)是遗忘门的输入到隐藏层权重。

+   ![](img/f2d2a796-703e-4706-92b1-4e997698d77c.png)是遗忘门的隐藏到隐藏层权重。

+   ![](img/5ced3ee9-3893-49af-8838-d6cb212cae67.png)是遗忘门的偏置。

以下图示显示了遗忘门。正如你所见，输入![](img/c3f4ca14-e495-43c9-b9ca-27920eabfac7.png)与![](img/777883d9-25a8-472b-81d7-453ad559c9a3.png)以及上一个隐藏状态![](img/d9a8a349-d2d9-4691-9599-d96500320a27.png)与![](img/c1624be1-c68a-4969-b7a0-f8bc9dc24888.png)相乘，然后两者相加并发送到 sigmoid 函数，返回![](img/0685de1f-7e90-471e-a062-9d42c26055bb.png)，如下所示：

![](img/21c2e1af-c75a-466b-8f67-e4226799484f.png)

# 输入门

输入门负责决定应该存储在细胞状态中的信息。让我们考虑相同的例子：

*Harry is a good singer. He lives in New York. Zayn is also a good singer.*

在遗忘门从细胞状态中移除信息后，输入门决定保留在记忆中的信息。在这里，由于遗忘门已从细胞状态中移除了关于 Harry 的信息，输入门决定用*Zayn*的信息更新细胞状态。

类似于遗忘门，输入门由 sigmoid 函数控制，返回 0 到 1 范围内的输出。如果返回`1`，则特定信息将被存储/更新到细胞状态中，如果返回`0`，则不会将信息存储到细胞状态中。在时间步![](img/b94192a5-ca93-4dbf-a997-9554df7b2123.png)，输入门![](img/5dc8f765-46d8-4432-93c9-db97e4a66144.png)的表达如下：

![](img/33badc8d-062a-47a8-acce-0efff9475568.png)

在这里，适用以下内容：

+   ![](img/1e52f1a4-475a-4814-8d88-10d81da859ab.png)是输入门的输入到隐藏层权重。

+   ![](img/68924da4-5fb6-4342-ac3f-4a309454abc4.png)是输入门的隐藏到隐藏权重。

+   ![](img/8d4321fe-6354-4f24-90dd-92cd3a1e5211.png)是输入门的偏置。

以下图示显示了输入门：

![](img/2eaad551-d7b1-4fdd-a0bc-954e90830f75.png)

# 输出门

我们将在细胞状态（记忆）中有大量信息。输出门负责决定从细胞状态中取出哪些信息作为输出。考虑以下句子：

*赞恩的首张专辑取得了巨大成功。祝贺 ____*。

输出门将查找细胞状态中的所有信息，并选择正确的信息填补空白。在这里，`congrats` 是用来描述名词的形容词。因此，输出门将预测填补空白的是 *Zayn*（名词）。与其他门类似，它也由一个 sigmoid 函数控制。在时间步![](img/187722bf-21fb-4d17-8dde-39169fa5ac77.png)，输出门 ![](img/0655737c-c309-48d8-ba3c-32ad669ea0bf.png) 的表示如下：

![](img/bdc58381-1789-40f8-90e4-d975b57d2cd7.png)

在这里，适用以下规则：

+   ![](img/b76a5820-fcd7-4782-83ee-1dfdd24a3ca0.png)是输出门的输入到隐藏层权重

+   ![](img/50f5fe09-5490-45c5-9c4d-9d939cad28ad.png)是输出门的隐藏到隐藏层权重

+   ![](img/d6cf8ce6-6d74-442f-b5e9-f8ec08c0dbb8.png)是输出门的输入到隐藏层权重

输出门如下图所示：

![](img/b3abb122-fc53-43d3-9e09-6f53d8e6e370.png)是输出门的偏置

# 更新细胞状态

我们刚刚学习了 LSTM 网络中所有三个门是如何工作的，但问题是，我们如何实际通过门来更新细胞状态，添加相关的新信息并删除不需要的信息？

首先，我们将看看如何向细胞状态添加新的相关信息。为了容纳可以添加到细胞状态（记忆）中的所有新信息，我们创建一个新的向量称为 ![](img/7010184a-9902-4f50-b3dc-64ff49f06e92.png)。它被称为 **候选状态** 或 **内部状态向量**。与由 sigmoid 函数调节的门不同，候选状态由 tanh 函数调节，但为什么呢？Sigmoid 函数返回范围在 `0` 到 `1` 之间的值，即始终为正。我们需要允许 ![](img/0693ff3d-4f5e-4004-8865-614ee2553802.png) 的值可以是正或负。因此，我们使用 tanh 函数，它返回范围在 `-1` 到 `+1` 之间的值。

在时间步![](img/f553621f-c7c9-4f3d-bd57-674f16d5b2e5.png)，候选状态 ![](img/c3f66b9a-6c7e-4fbb-9a00-6e6e9643b218.png) 的表示如下：

![](img/fd5e300f-79f4-480c-92b5-7a53392442e4.png)

在这里，适用以下规则：

+   ![](img/a6aed5e2-ec9f-4077-9175-341c5adf258b.png)是候选状态的输入到隐藏层权重

+   ![](img/d96cb484-7d53-4b0b-a81a-c3d05f1ba252.png)是候选状态的隐藏到隐藏层权重

+   ![](img/e39f7759-421a-4c62-af06-c6ba16fc8c7f.png)是候选状态的偏置

因此，候选状态包含了所有可以添加到细胞状态（记忆）中的新信息。下图显示了候选状态：

![](img/31a44855-b4bb-4697-9d1b-43e46f81c937.png)

我们如何决定候选状态中的信息是否相关？我们如何决定是否将候选状态中的新信息添加到细胞状态中？我们学到，输入门负责决定是否添加新信息，因此如果我们将 ![](img/69f0d59a-7e8f-47bb-a52b-c954f49fd889.png) 和 ![](img/31980fd0-d617-4a69-9dd7-3c72700ba834.png) 相乘，我们只获得应添加到记忆中的相关信息。

换句话说，我们知道如果信息不需要，则输入门返回 0，如果信息需要则返回 1。例如 ![](img/f1b28985-aaf5-4103-bf2b-582c9929da87.png)，那么将 ![](img/50c4bd55-f5c8-49ab-bd1f-a2ee36695ba9.png) 和 ![](img/83b75b28-d035-496c-8ff6-f30d77937811.png) 相乘得到 0，这意味着 ![](img/824cdab3-ee4c-40e6-b090-ce2faabd730e.png) 中的信息不需要更新细胞状态的 ![](img/6a312070-52f1-46dc-9a3d-05455a03adc0.png)。当 ![](img/8dc8a573-bc68-47a1-aa12-294c68c36324.png) 时，将 ![](img/bec8dc05-a2ad-403c-a9c6-69d783388261.png) 和 ![](img/fe5e1426-2690-4cb9-9f00-954ed0822831.png) 相乘得到 ![](img/a3f46a0c-153c-474b-a988-eb76ebb2a079.png)，这意味着我们可以更新 ![](img/1a5bd9a1-15bc-4e57-a25f-7a72def34a41.png) 中的信息到细胞状态。

将新信息添加到细胞状态的输入门 ![](img/6e2dd534-874f-40da-b9e3-88440ae9c253.png) 和候选状态 ![](img/5904f3a8-5a43-49d1-9ce0-71ebee730515.png) 在下图中显示：

![](img/5d8b922a-ead9-4d15-927e-6dcc229f8532.png)

现在，我们将看到如何从先前的细胞状态中移除不再需要的信息。

我们学到，遗忘门用于删除细胞状态中不需要的信息。因此，如果我们将先前的细胞状态 ![](img/41f210c4-6880-4e4e-90aa-dee67fd723d6.png) 和遗忘门 ![](img/f52ba505-fe3d-49f2-99a6-0df0238b26a4.png) 相乘，那么我们只保留细胞状态中的相关信息。

例如 ![](img/08a5d23e-261b-4cb2-a99f-db63239c239f.png)，那么将 ![](img/a9107933-985a-4348-a5d5-bd272786ffe3.png) 和 ![](img/bd8a9753-06d7-4593-aba6-b51903567a26.png) 相乘得到 0，这意味着细胞状态中的信息 ![](img/0df3abae-8597-478a-875f-89dd39594f57.png) 不需要，并且应该被移除（遗忘）。当 ![](img/50fa7006-ccca-4b1a-b53b-c3b259b3549f.png) 时，将 ![](img/0babfbf6-486a-4e70-b752-5e7009839fc6.png) 和 ![](img/541215b2-01bf-47eb-bc99-0e1c4226b0d1.png) 相乘得到 ![](img/4d95c71f-bcb4-455a-8245-aa6ba10e7237.png)，这意味着先前细胞状态中的信息是必需的，不应被移除。

使用遗忘门 ![](img/41c44951-0a3a-44e8-9671-34acc507a4ad.png) 从先前的细胞状态 ![](img/ed140ab2-3961-4752-a6b3-c6a0b061939d.png) 中移除信息如下图所示：

![](img/be393d9d-33f7-429d-aeca-a2db2e89517e.png)

因此，简言之，我们通过乘以 ![](img/8aacb5a6-fc50-4376-bfa3-d7bd6a6be88f.png) 和 ![](img/108eb432-d875-4f0d-91d8-bfbbb53606c2.png) 来添加新信息，乘以 **![](img/7d32cace-bc58-4661-aa73-3fa46a6edaa5.png)** 和 ![](img/e7680c9e-a545-4bc4-9fd5-44215a5ed1eb.png) 来移除信息，从而更新我们的细胞状态。我们可以将细胞状态方程表达如下：

![](img/df6266fe-ac8b-4679-90fa-3c23aee2a14d.png)

# 更新隐藏状态

我们刚刚学习了如何更新细胞状态中的信息。现在，我们将看到如何更新隐藏状态中的信息。我们知道隐藏状态 ![](img/09be6d36-046f-4f1c-aac7-be503fe678fb.png) 用于计算输出，但是我们如何计算输出呢？

我们知道输出门负责决定从细胞状态中获取什么信息作为输出。因此，将 ![](img/7c5e428a-dd2c-44ec-9839-6f95408c2396.png) 和细胞状态 ![](img/0a08740e-e7f3-49de-b21f-3d13898c2c65.png) 的双曲正切乘积（将其压缩在 -1 和 +1 之间）给出输出。

因此，隐藏状态 ![](img/b9849aba-ed4b-4a83-97d7-544790335046.png)，可以表达如下：

![](img/074a4b48-cf46-4633-996d-68c95dbf1c1c.png)

下图显示了隐藏状态 ![](img/fecd1eda-5aa0-43c3-b37b-a46c9a50d939.png) 如何通过乘以 ![](img/0753373c-df5f-4009-817a-6742f853390a.png) 和 ![](img/0c969a7e-719c-4526-8979-865b3bec2bdf.png) 来计算：

![](img/045de488-09e7-4920-9b96-2a2aca3107f0.png)

最后，一旦我们有了隐藏状态值，我们可以应用 softmax 函数并计算 ![](img/95012735-9e45-42d4-93a2-5d1732452c9b.png) 如下：

![](img/ab5ba5f9-a9a3-42aa-abff-75f6600f1739.png)

这里，![](img/fa68cd80-0812-475e-82b0-c8313bf3da60.png) 是隐藏到输出层的权重。

# LSTM 中的前向传播

将所有部分整合起来，所有操作后的最终 LSTM 细胞显示在以下图表中。 细胞状态和隐藏状态在时间步骤间共享，这意味着 LSTM 在时间步骤 ![](img/5e3b3d43-c481-4df0-a75e-8367d659a265.png) 计算细胞状态 ![](img/1fbe13e7-ac1c-4577-aba8-10213738f557.png) 和隐藏状态 ![](img/a41448bd-7eaa-4c55-af63-23f53b072f96.png)，并将其发送到下一个时间步骤：

![](img/48f7218f-01e9-4109-beb2-3f2449f08570.png)

LSTM 细胞中完整的前向传播步骤如下所示：

1.  **输入门**：![](img/cea4e9e7-fb6b-4e0d-9ae7-34e059e6e018.png)

1.  **遗忘门**：![](img/ae23d56d-86db-4dfc-9ffd-a769bfef5a83.png)

1.  **输出门**：![](img/24d931a3-a0d3-4bfd-9c7f-4f1cae0c3e7f.png)

1.  **候选状态**：![](img/371519ee-9035-417e-b657-08ee8e7fc7f1.png)

1.  **细胞状态**：![](img/5a4c735f-b059-4e30-bfd5-0a3436436082.png)

1.  **隐藏状态**：![](img/0aea9e04-47b6-4d68-a0e5-7ea94594c929.png)

1.  **输出**：![](img/9c7f33b3-1b3b-470c-af11-451d8d885391.png)

# LSTM 中的反向传播

我们在每个时间步计算损失，以确定我们的 LSTM 模型预测输出的好坏。假设我们使用交叉熵作为损失函数，则在时间步 ![](img/877e5abb-f731-421a-a531-fee33027608d.png) 的损失 ![](img/96cbb175-2daf-4cc9-bfab-ff5fcdf1b041.png) 如下方程所示：

![](img/f1e29a10-9192-4780-a6b7-0521d3c4974c.png)

这里，![](img/8df65662-e615-422c-ae9d-593b19d4e085.png) 是实际输出，![](img/d8c35c2e-3ac8-4ed5-b837-88c02616c79d.png) 是时间步 ![](img/e5c01070-a571-4262-b25c-329c0a2ea24a.png) 的预测输出。

我们的最终损失是所有时间步的损失之和，可以表示为：

![](img/e12ed7eb-f381-4809-a198-5989df1f6bc3.png)

我们使用梯度下降来最小化损失。我们找到损失对网络中所有权重的导数，并找到最优权重以最小化损失：

+   我们有四个输入到隐藏层的权重，![](img/57a0c8ec-248e-4ae1-b055-3781a8e24024.png)，分别是输入门、遗忘门、输出门和候选状态的输入到隐藏层权重。

+   我们有四个隐藏到隐藏层的权重，![](img/2c7f7639-7eed-4913-89be-650cf0771aba.png)，分别对应输入门、遗忘门、输出门和候选状态的隐藏到隐藏层权重。

+   我们有一个隐藏到输出层的权重，![](img/594a46af-c525-4fc2-bbda-963f90fe849b.png)

我们通过梯度下降找到所有这些权重的最优值，并根据权重更新规则更新权重。权重更新规则如下方程所示：

![](img/3afc2bbe-65d5-463e-a631-aa647fd0892f.png)

在下一节中，我们将逐步查看如何计算 LSTM 单元中所有权重相对于损失的梯度。

如果你对推导所有权重的梯度不感兴趣，可以跳过即将到来的部分。然而，这将加强你对 LSTM 单元的理解。

# 相对于门的梯度

计算 LSTM 单元中所有权重相对于损失的梯度需要计算所有门和候选状态的梯度。因此，在本节中，我们将学习如何计算损失函数相对于所有门和候选状态的梯度。

在我们开始之前，让我们回顾以下两件事：

+   sigmoid 函数的导数表达如下：

![](img/86b0543e-e6c3-4ad4-b3a2-eef76bbb11e8.png)

+   tanh 函数的导数表达如下：

![](img/03c9db62-92a2-4bd3-ba0a-e15f68d3269c.png)

在即将进行的计算中，我们将在多个地方使用损失相对于隐藏状态 ![](img/04b1bbe0-9a98-4fd2-bf43-2326ca721944.png) 和细胞状态 **![](img/79947213-05ea-4838-ab92-d50b5c63ffa1.png)** 的梯度。因此，首先，我们将看看如何计算损失相对于隐藏状态 ![](img/5a7453ed-c6b2-4601-9da4-b754deab49f9.png) 和细胞状态 **![](img/cb33785c-773a-484f-a66e-a49ce32caecb.png)** 的梯度。

首先，让我们看看如何计算**损失相对于隐藏状态的梯度**，![](img/aff235e2-6d45-4099-b535-b73f298c37aa.png)。

我们知道输出 ![](img/0c5c515b-482d-44a0-bc55-7cc367e382c5.png) 的计算如下：

![](img/c505c7c0-8f83-463a-a971-555a168a4ea6.png)

假设 ![](img/3e49322e-fb07-4005-b193-3dbf16d74314.png)。我们在 ![](img/5b3182c0-4ae1-4624-870a-62733e90f554.png) 中有 ![](img/680eac77-b21b-4b96-8f6a-f349834419eb.png) 项，因此根据链式法则，我们可以写出以下内容：

![](img/e2808e7a-0d8e-42c3-a9be-2b70221debf8.png)

![](img/88e76828-1e21-44e2-8f47-03e0dd1d8128.png)

我们已经看到如何在 第四章，*使用 RNN 生成歌词* 中计算 ![](img/849bb929-fbf6-4063-80d4-e45bd1383e47.png)，因此直接从 第四章 的方程 *(9)* 中，*使用 RNN 生成歌词*，我们可以写出以下内容：

![](img/bd0df0a9-66e1-4f63-b928-38eeacd4d190.png)

现在，让我们看看如何计算**损失相对于细胞状态的梯度**，![](img/45208875-14c6-4221-b079-f1bca82c3913.png)。

要计算损失相对于细胞状态的梯度，请查看前向传播的方程，并找出哪个方程中有 ![](img/1812c1f1-9e31-45ac-8c67-4c1958b3f80e.png) 项。在隐藏状态的方程中，我们有如下的 ![](img/87e235e5-bc30-41c6-a638-0271e6d2094e.png) 项：

![](img/2d5f92d3-a3b1-4bc9-96a9-65bfe468d4a1.png)

因此，根据链式法则，我们可以写出以下内容：

![](img/11f7f1d7-0339-44d8-a188-a9e19b3f9899.png)

我们知道 tanh 的导数是 ![](img/85b44914-9957-44df-9cca-f1d59de773d6.png)，因此我们可以写出以下内容：

![](img/dbfceaaf-2494-47d5-af80-24451621f59e.png)

现在我们已经计算出损失相对于隐藏状态和细胞状态的梯度，让我们看看如何逐个计算损失相对于所有门的梯度。

首先，我们将看看如何计算**损失相对于输出门的梯度**，![](img/829eda1c-c23c-4aa1-b061-324a07987dab.png)。

要计算损失相对于输出门的梯度，请查看前向传播的方程，并找出哪个方程中有 ![](img/3868b0a2-4822-49e2-b864-02ffca131759.png) 项。在隐藏状态的方程中，我们有如下的 ![](img/2bfbcba3-17e5-4089-84a9-1d8b8efb8687.png) 项：

![](img/fd002aac-4b10-42a1-be7d-e82b88769eb0.png)

因此，根据链式法则，我们可以写出以下内容：

![](img/39d60837-5619-4502-990c-f3e63598109c.png)

![](img/144189a8-3f1f-4ab2-bec2-bcf4ed72cc47.png)

现在我们将看到如何计算对输入门的损失梯度，![](img/221da6b9-eecb-44c9-8b01-48419b895c14.png)。

我们在细胞状态方程中有![](img/426327c3-f651-46a0-9687-1de1676aabe9.png)项用于![](img/29481156-1f9e-48fe-93a8-e6378d53aad7.png)：

![](img/2180062d-8ea8-4049-a8bd-d59742ef4f45.png)

根据链式法则，我们可以写成以下形式：

![](img/2bc7fdbf-45c6-4f4a-a9e0-7be079d6cfa2.png)

![](img/2b7f7ac9-0f08-48d3-9ffe-9aee4389eb2e.png)

现在我们学习如何计算对遗忘门的损失梯度，![](img/5a957198-b737-4612-aaf0-dda6c302fae7.png)。

我们还在细胞状态方程中有![](img/2407ceae-de39-4719-9d81-aefdae7865cf.png)项用于![](img/d6f6908f-c0a5-47fc-8ee5-ff26b6648737.png)：

![](img/7cf851c6-598f-404b-bfc2-5a5c5b719791.png)

根据链式法则，我们可以写成以下形式：

![](img/fa0defe6-30a7-4a37-9e6c-3e7fb8bc18e6.png)

![](img/3a9a642e-f43e-4d53-9c52-813404648bb3.png)

最后，我们学习如何计算对候选状态的损失梯度，![](img/72970edd-bb07-4b02-b1e8-34d6f1a99f5f.png)。

我们还在细胞状态方程中有![](img/f39da3ff-6660-4c2f-b180-dc541d98a818.png)项用于![](img/51ae1b17-9ae6-494f-82bb-00766ee993e5.png)：

![](img/62b91a3a-5975-4d77-9fdc-25e35e59dd63.png)

因此，根据链式法则，我们可以写成以下形式：

![](img/837534e6-29a8-49fc-b25e-eeef4ba43fdb.png)

![](img/0ed71810-1be0-4191-a7a2-5b00efb2d45b.png)

因此，我们已经计算出了损失对所有门和候选状态的梯度。在接下来的部分中，我们将看到如何计算损失对 LSTM 单元中所有权重的梯度。

# 对权重的梯度

现在让我们看看如何计算损失对 LSTM 单元中所有权重的梯度。

# 对 V 的梯度

在预测输出![](img/7edada70-d300-4872-811f-57e83ea23e73.png)后，我们处于网络的最后一层。因为我们在进行反向传播，即从输出层到输入层，我们的第一个权重将是隐藏到输出层的权重，![](img/28e1042c-b857-4de6-a4e4-ebe2576eaad0.png)。

我们一直学到最后的损失是所有时间步长的损失之和。类似地，我们最终的梯度是所有时间步骤的梯度之和，如下所示：

![](img/f644de61-9b4b-4dd2-8260-d7900dd7387d.png)

如果我们有![](img/d2193d10-a298-4be5-a7de-5de99b7e6253.png)层，那么我们可以将损失对![](img/fd9b1b5c-6a97-4fbd-be40-6046235b7b27.png)的梯度写成如下形式：

![](img/4706bc36-2544-43fa-b56a-9ce238a09a4a.png)

由于 LSTM 的最终方程式，即![](img/38547eb6-31a6-4683-8d02-dbfd3415c3a7.png)，与 RNN 相同，计算与![](img/941c7ac5-91ab-420b-b793-8ff4e11f5d2d.png)相关的损失梯度与我们在 RNN 中计算的完全相同。因此，我们可以直接写出以下内容：

![](img/8b2f7164-c2b8-4f1a-8d17-26eb7117039e.png)

# 与 W 相关的梯度

现在我们将看看如何计算隐藏到隐藏层权重![](img/4dde845e-01a8-4703-9bce-6c15a8f049bb.png)对所有门和候选状态的损失梯度。

让我们计算**与![](img/df71f656-1289-49e5-a2e9-8ca49ea63d3d.png)相关的损失梯度**。

回想一下输入门的方程式，如下所示：

![](img/e28e8482-bf06-4595-8ef5-eab0ace4adb6.png)

因此，根据链式法则，我们可以写出以下内容：

![](img/32367a92-eb50-421a-9b6a-86fc3a43c61e.png)

让我们计算前述方程中的每一项。

我们已经看到如何计算第一项，即损失关于输入门的梯度，![](img/6640ea99-0003-480a-8009-9061d15a2c17.png)，在*门梯度*部分。参考方程式(*2*)。

因此，让我们看看第二项：

![](img/1cbcee6f-9c1e-4f01-a863-8b173f8b209b.png)

由于我们知道 sigmoid 函数的导数，即![](img/8ecc5286-7a2f-41f3-9937-5aca92701106.png)，因此我们可以写出以下内容：

![](img/9bb15bec-f3c9-4df2-b5be-03b87e066c27.png)

但是![](img/3688d455-05da-4c1c-ace6-2f0fd93bcd95.png)已经是 sigmoid 的结果，即![](img/e041ff53-d1f5-4b8a-b2b0-dd882b417807.png)，因此我们可以直接写出![](img/98e86e32-88ed-41c3-ab2e-e61464eed7d5.png)，因此，我们的方程式变为以下内容：

![](img/2cc5def6-ee4e-477e-9caa-cde7c207c31c.png)

因此，我们计算损失关于![](img/c26c4547-7431-4072-acf1-906ab26665f0.png)梯度的最终方程变为以下内容：

![](img/6ff4b581-a640-45a5-a60d-eef08c056f74.png)

现在，让我们找出与![](img/4ca82c19-eca4-49d7-8179-e107c10b80b8.png)相关的**损失梯度**。

回想一下遗忘门的方程式，如下所示：

![](img/dd4f5c88-16ad-4dfb-a8f2-25f851999326.png)

因此，根据链式法则，我们可以写出以下内容：

![](img/e8455693-3b32-4af2-9b88-3a35d2e33b9e.png)

我们已经看到如何在门梯度部分计算![](img/079e755e-0b0f-4a20-bb9b-5f93b62f8679.png)。参考方程式*(3)*。因此，让我们看看计算第二项：

![](img/2b51c294-be13-4f6c-bffc-994a3f9b0b75.png)

因此，我们计算梯度损失与![](img/f83344db-5707-4cc7-a86a-ea1576fdffaa.png)相关的最终方程式如下：

![](img/51f8ec69-c2b4-4bfe-8d74-da0c49e333bc.png)

让我们计算**与![](img/76bccc57-edc1-4e61-9580-a535b74fb0f8.png)相关的损失梯度**。

回想一下输出门的方程式，如下所示：

![](img/b43d3c0c-cfce-4de7-b3c3-cac620dcc4d7.png)

因此，使用链式法则，我们可以写成以下形式：

![](img/4a839211-1e76-4573-8f73-c8746085b65b.png)

检查方程式 *(1)* 的第一项。第二项可以计算如下：

![](img/487f7a94-71e3-48f0-9aae-bf081c5029b5.png)

因此，我们计算损失相对于 ![](img/36529b85-5e3a-4c49-9a09-8b39799f1406.png) 梯度的最终方程式如下：

![](img/90cf0690-88f4-4baf-91ef-6a650ed3306b.png)

让我们继续计算**相对于** ![](img/645a2ad4-94f6-4fa9-a97c-dc7086d89783.png) 的梯度。

回想一下候选状态方程式：

![](img/788211a7-312f-4f5c-a35d-ae8fbf5019b9.png)

因此，使用链式法则，我们可以写成以下形式：

![](img/fadea06d-8414-447f-b135-3ad8bd773de8.png)

参考方程 *(4)* 的第一项。第二项可以计算如下：

![](img/6a723bc0-48b1-4c5b-ba17-788449f6e3eb.png)

我们知道 tanh 的导数是 ![](img/45d023b1-640a-4f3f-83b0-c88c7e00a6f9.png)，因此我们可以写成以下形式：

![](img/6c747d9d-b05c-4db0-8065-c11c54a645fd.png)

因此，我们计算损失相对于 ![](img/2a8030d8-33f7-445b-83e4-5015eccd81fc.png) 的梯度的最终方程式如下：

![](img/389d4034-18d8-4a33-8b45-54addca5326b.png)

# 关于 U 的梯度

让我们计算损失相对于隐藏到输入层权重 ![](img/49138555-3ac0-4529-9db6-d52f3739c051.png) 对所有门和候选状态的梯度。相对于 ![](img/49138555-3ac0-4529-9db6-d52f3739c051.png) 的损失梯度计算与我们相对于 ![](img/d46ef0c9-23cb-49cf-820a-0bfa688f8c12.png) 计算的梯度完全相同，只是最后一项是 ![](img/aceb813e-47ff-40b6-8da5-70db589b339e.png) 而不是 ![](img/326b737b-9984-4c3c-88f2-13f19d801669.png)。让我们详细探讨一下这是什么意思。

让我们找出**相对于** ![](img/b463fa17-3ede-4bf3-b562-4c19ef4e11ad.png) 的梯度。

输入门的方程式如下：

![](img/592ff201-63c2-44bd-b9d3-a42e4ee5e0e0.png)

因此，使用链式法则，我们可以写成以下形式：

![](img/c88adacc-0b75-45fc-9bec-534c9af7073b.png)

让我们计算前述方程式中的每一项。我们已经从方程 *(2)* 知道了第一项。因此，第二项可以计算如下：

![](img/6824b2e1-a042-499c-91d8-8fb31bd88179.png)

因此，我们计算损失相对于 ![](img/42d1129a-5f98-42a9-a85f-17fd6f8c092a.png) 梯度的最终方程式如下：

![](img/8030e7d5-2f29-4b81-8914-362627364f7b.png)

正如您所看到的，前述方程式与 ![](img/a1abf315-812d-49bf-ac5f-5306b5ad711f.png) 完全相同，只是最后一项是 ![](img/f52d64fe-0324-4fb3-9ef8-79270de4df65.png) 而不是 ![](img/954523a4-ac71-4553-ab87-971814fddc4b.png)。对于所有其他权重也适用，因此我们可以直接写出方程式如下：

+   **相对于** ![](img/7f403a62-a89d-432d-9e57-c100f6450e58.png) 的梯度：

![](img/531be703-bc6f-462d-8173-20542d359a9c.png)

+   损失关于![](img/ddf83867-c5d5-4321-98c2-ea00df20d9b1.png)的梯度：

![](img/62ee2e00-65f1-4596-acb5-26ef6bff42e5.png)

+   损失关于![](img/54698652-aa77-456c-aa2f-cf02bdcd93c7.png)的梯度：

![](img/e42a96e1-641a-4e70-9c22-7aaf41893047.png)

计算梯度后，针对所有这些权重，我们使用权重更新规则更新它们，并最小化损失。

# 使用 LSTM 模型预测比特币价格

我们已经了解到 LSTM 模型广泛用于序列数据集，即有序的数据集。在本节中，我们将学习如何使用 LSTM 网络进行时间序列分析。我们将学习如何使用 LSTM 网络预测比特币价格。

首先，我们按如下方式导入所需的库：

```py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
%matplotlib inline 
plt.style.use('ggplot')

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import warnings
warnings.filterwarnings('ignore')
```

# 数据准备

现在，我们将看到如何准备我们的数据集，以便我们的 LSTM 网络需要。首先，我们按如下方式读取输入数据集：

```py
df = pd.read_csv('data/btc.csv')
```

然后我们展示数据集的几行：

```py
df.head()
```

上述代码生成如下输出：

![](img/482c8f90-ae85-4b04-987d-7a0832ef3cd3.png)

如前述数据框所示，`Close`列表示比特币的收盘价。我们只需要`Close`列来进行预测，因此我们只取该特定列：

```py
data = df['Close'].values
```

接下来，我们标准化数据并将其缩放到相同的尺度：

```py
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1, 1))
```

然后，我们绘制并观察比特币价格变化的趋势。由于我们缩放了价格，它不是一个很大的数值：

```py
plt.plot(data)
plt.xlabel('Days')
plt.ylabel('Price')
plt.grid()
```

生成如下图所示的绘图：

![](img/cb8b7244-692d-4386-8014-3db1d5a5a4af.png)

现在，我们定义一个称为`get_data`函数的函数，它生成输入和输出。它以数据和`window_size`作为输入，并生成输入和目标列。

这里的窗口大小是多少？我们将*x*值向前移动`window_size`次，并得到*y*值。例如，如下表所示，当`window_size`等于 1 时，*y*值恰好比*x*值提前一步：

| **x** | **y** |
| --- | --- |
| 0.13 | 0.56 |
| 0.56 | 0.11 |
| 0.11 | 0.40 |
| 0.40 | 0.63 |

函数`get_data()`的定义如下：

```py
def get_data(data, window_size):
    X = []
    y = []

    i = 0

    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])

        i += 1
    assert len(X) == len(y)
    return X, y
```

我们选择`window_size`为`7`并生成输入和输出：

```py
X, y = get_data(data, window_size = 7)
```

将前`1000`个点视为训练集，其余点视为测试集：

```py
#train set
X_train = np.array(X[:1000])
y_train = np.array(y[:1000])

#test set
X_test = np.array(X[1000:])
y_test = np.array(y[1000:])
```

`X_train`的形状如下所示：

```py
X_train.shape

(1000,7,1)
```

前面的形状表示什么？它意味着`sample_size`、`time_steps`和`features`函数及 LSTM 网络需要的输入正是如下所示：

+   `1000`设置数据点数目（`sample_size`）

+   `7`指定窗口大小（`time_steps`）

+   `1`指定我们数据集的维度（`features`）

# 定义参数

定义网络参数如下：

```py
batch_size = 7
window_size = 7
hidden_layer = 256
learning_rate = 0.001
```

为我们的输入和输出定义占位符：

```py
input = tf.placeholder(tf.float32, [batch_size, window_size, 1])
target = tf.placeholder(tf.float32, [batch_size, 1])
```

现在，让我们定义我们在 LSTM 单元中将使用的所有权重。

输入门的权重定义如下：

```py
U_i = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
W_i = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
b_i = tf.Variable(tf.zeros([hidden_layer]))
```

忘记门的权重定义如下：

```py
U_f = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
W_f = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
b_f = tf.Variable(tf.zeros([hidden_layer]))
```

输出门的权重定义如下：

```py
U_o = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
W_o = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
b_o = tf.Variable(tf.zeros([hidden_layer]))
```

候选状态的权重定义如下：

```py
U_g = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
W_g = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
b_g = tf.Variable(tf.zeros([hidden_layer]))
```

输出层的权重如下所示：

```py
V = tf.Variable(tf.truncated_normal([hidden_layer, 1], stddev=0.05))
b_v = tf.Variable(tf.zeros([1]))
```

# 定义 LSTM 单元

现在，我们定义名为`LSTM_cell`的函数，它将细胞状态和隐藏状态作为输出返回。回顾我们在 LSTM 前向传播中看到的步骤，它的实现如下所示。`LSTM_cell`接受输入、先前隐藏状态和先前细胞状态作为输入，并返回当前细胞状态和当前隐藏状态作为输出：

```py
def LSTM_cell(input, prev_hidden_state, prev_cell_state):

    it = tf.sigmoid(tf.matmul(input, U_i) + tf.matmul(prev_hidden_state, W_i) + b_i)

    ft = tf.sigmoid(tf.matmul(input, U_f) + tf.matmul(prev_hidden_state, W_f) + b_f)

    ot = tf.sigmoid(tf.matmul(input, U_o) + tf.matmul(prev_hidden_state, W_o) + b_o)

    gt = tf.tanh(tf.matmul(input, U_g) + tf.matmul(prev_hidden_state, W_g) + b_g)

    ct = (prev_cell_state * ft) + (it * gt)

    ht = ot * tf.tanh(ct)

    return ct, ht
```

# 定义前向传播

现在我们将执行前向传播并预测输出，![](img/1e63eb13-17e9-4d49-b4da-5f050305776d.png)，并初始化一个名为`y_hat`的列表以存储输出：

```py
y_hat = []
```

对于每次迭代，我们计算输出并将其存储在`y_hat`列表中：

```py
for i in range(batch_size):
```

我们初始化隐藏状态和细胞状态：

```py
    hidden_state = np.zeros([1, hidden_layer], dtype=np.float32) 
    cell_state = np.zeros([1, hidden_layer], dtype=np.float32)
```

我们执行前向传播，并计算每个时间步长的 LSTM 单元的隐藏状态和细胞状态：

```py
    for t in range(window_size):
        cell_state, hidden_state = LSTM_cell(tf.reshape(input[i][t], (-1, 1)), hidden_state, cell_state)
```

我们知道输出 ![](img/8cdf455d-e02b-4e47-b962-438e508ee1ff.png) 可以计算如下：

![](img/214a7de4-6b0e-4c5c-8fd2-c9c8ec5ad246.png)

计算`y_hat`，并将其附加到`y_hat`列表中：

```py
    y_hat.append(tf.matmul(hidden_state, V) + b_v)
```

# 定义反向传播

在执行前向传播并预测输出之后，我们计算损失。我们使用均方误差作为我们的损失函数，总损失是所有时间步长上损失的总和：

```py
losses = []

for i in range(len(y_hat)):
    losses.append(tf.losses.mean_squared_error(tf.reshape(target[i], (-1, 1)), y_hat[i]))

loss = tf.reduce_mean(losses)
```

为了避免梯度爆炸问题，我们执行梯度裁剪：

```py
gradients = tf.gradients(loss, tf.trainable_variables())
clipped, _ = tf.clip_by_global_norm(gradients, 4.0)
```

我们使用 Adam 优化器并最小化我们的损失函数：

```py
optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(gradients, tf.trainable_variables()))
```

# 训练 LSTM 模型

开始 TensorFlow 会话并初始化所有变量：

```py
session = tf.Session()
session.run(tf.global_variables_initializer())
```

设置`epochs`的数量：

```py
epochs = 100
```

然后，对于每次迭代，执行以下操作：

```py
for i in range(epochs):

    train_predictions = []
    index = 0
    epoch_loss = []
```

然后对数据批次进行抽样并训练网络：

```py
    while(index + batch_size) <= len(X_train):

        X_batch = X_train[index:index+batch_size]
        y_batch = y_train[index:index+batch_size]

        #predict the price and compute the loss
        predicted, loss_val, _ = session.run([y_hat, loss, optimizer], feed_dict={input:X_batch, target:y_batch})

        #store the loss in the epoch_loss list
        epoch_loss.append(loss_val)

        #store the predictions in the train_predictions list
        train_predictions.append(predicted)
        index += batch_size
```

在每`10`次迭代上打印损失：

```py
     if (i % 10)== 0:
        print 'Epoch {}, Loss: {} '.format(i,np.mean(epoch_loss))
```

正如您可以在以下输出中看到的，损失随着 epochs 的增加而减少：

```py
Epoch 0, Loss: 0.0402321927249 
Epoch 10, Loss: 0.0244581680745 
Epoch 20, Loss: 0.0177710317075 
Epoch 30, Loss: 0.0117778982967 
Epoch 40, Loss: 0.00901956297457 
Epoch 50, Loss: 0.0112476013601 
Epoch 60, Loss: 0.00944950990379 
Epoch 70, Loss: 0.00822851061821 
Epoch 80, Loss: 0.00766260037199 
Epoch 90, Loss: 0.00710930628702 
```

# 使用 LSTM 模型进行预测

现在我们将开始对测试集进行预测：

```py
predicted_output = []
i = 0
while i+batch_size <= len(X_test): 

    output = session.run([y_hat],feed_dict={input:X_test[i:i+batch_size]})
    i += batch_size
    predicted_output.append(output)
```

打印预测输出：

```py
predicted_output[0]
```

我们将得到如下结果：

```py
[[array([[-0.60426176]], dtype=float32),
  array([[-0.60155034]], dtype=float32),
  array([[-0.60079575]], dtype=float32),
  array([[-0.599668]], dtype=float32),
  array([[-0.5991149]], dtype=float32),
  array([[-0.6008351]], dtype=float32),
  array([[-0.5970466]], dtype=float32)]]
```

正如您所见，测试预测值的值是嵌套列表，因此我们将它们展开：

```py
predicted_values_test = []
for i in range(len(predicted_output)):
  for j in range(len(predicted_output[i][0])):
    predicted_values_test.append(predicted_output[i][0][j])
```

现在，如果我们打印预测值，它们不再是嵌套列表：

```py
predicted_values_test[0]

array([[-0.60426176]], dtype=float32)
```

由于我们将前`1000`个点作为训练集，我们对大于`1000`的时间步长进行预测：

```py
predictions = []
for i in range(1280):
      if i >= 1000:
        predictions.append(predicted_values_test[i-1019])
      else:
        predictions.append(None)
```

我们绘制并查看预测值与实际值的匹配程度：

```py
plt.figure(figsize=(16, 7))
plt.plot(data, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Price')
plt.grid()
plt.show()
```

正如您在以下图中所见，实际值显示为红色，预测值显示为蓝色。由于我们对大于`1000`时间步长进行预测，您可以看到在时间步骤**1000**之后，红色和蓝色线条彼此交错，这表明我们的模型正确预测了实际值：

![](img/2232e1fb-4b94-4d85-9ce8-fc94362edfc1.png)

# 门控循环单元

到目前为止，我们已经学习了 LSTM 单元如何使用不同的门，并解决了 RNN 的梯度消失问题。但是，正如您可能注意到的，由于存在许多门和状态，LSTM 单元具有太多的参数。

因此，在反向传播 LSTM 网络时，我们需要在每次迭代中更新大量参数。这增加了我们的训练时间。因此，我们引入了**门控循环单元（GRU）**，它作为 LSTM 单元的简化版本。与 LSTM 单元不同，GRU 单元只有两个门和一个隐藏状态。

RNN 中使用 GRU 单元如下图所示：

![](img/f5d48cdd-f7d0-48f1-99b2-f8820ea6616c.png)

# 理解 GRU 单元

如下图所示，GRU 单元只有两个门，称为更新门和重置门，以及一个隐藏状态：

![](img/e6de6de0-f01c-4f8a-a4cc-69467795b056.png)

让我们深入了解这些门是如何使用的，以及如何计算隐藏状态。

# 更新门

更新门有助于决定前一时间步![](img/ca543331-1f78-4714-8082-23c2df67e94a.png)的哪些信息可以传递到下一时间步![](img/40dc7154-c4d2-4617-9724-422b72a8ba16.png)。它基本上是输入门和遗忘门的组合，我们在 LSTM 单元中学到的内容。与 LSTM 单元的门类似，更新门也由 sigmoid 函数调节。

在时间步骤![](img/dbab5c15-c0fd-40dc-84d2-ed47b76524ac.png)，更新门![](img/f4a8957d-4e8a-4985-87d7-fc3a51192777.png)表达如下：

![](img/50ded757-84e8-4a7d-9c98-d3714c058f7d.png)

在这里，应用以下内容：

+   ![](img/685a5db9-7362-42dd-9150-e87553a756af.png)是更新门的输入到隐藏权重

+   ![](img/860a056f-7024-43a2-85e2-d723096fed6e.png)是更新门的隐藏到隐藏权重

+   ![](img/fc870073-2339-4aff-92de-88731221b785.png)是更新门的偏置

以下图示显示了更新门。如您所见，输入![](img/2148ff3c-2102-43cf-93be-d01300bb9d15.png)与![](img/4b197a14-642f-402d-ba45-ae45c3f3f7d2.png)相乘，并且先前的隐藏状态，![](img/23255254-0486-46a9-a27c-2d9289ffc00f.png)，0 和 1：

![](img/ea43579f-2493-4912-9181-43229cbbb46e.png)

# 重置门

重置门帮助决定如何将新信息添加到内存中，即它可以忘记多少过去信息。在时间步骤![](img/aff87a8a-21a9-4b42-8623-8fd73549c61f.png)，重置门![](img/dc243755-ac57-4e04-aaec-33f63ae1f369.png)表达如下：

![](img/1059b3f7-1f07-41c3-8607-2cd07a280781.png)

在这里，应用以下内容：

+   ![](img/663d529b-93df-49c5-9038-848d4758b1a9.png)是重置门的输入到隐藏权重

+   ![](img/5efc1bb2-5707-4aa5-93fb-ca29d9cbab72.png)是重置门的隐藏到隐藏权重

+   ![](img/5b8a4123-9c37-46a3-8371-b8e113c8aecc.png)是重置门的偏置

重置门如下图所示：

![](img/f58f7a88-7e0f-4107-975f-f7afe26e9a5d.png)

# 更新隐藏状态

我们刚刚学习了更新和重置门的工作原理，但这些门如何帮助更新隐藏状态呢？也就是说，如何利用重置门和更新门向隐藏状态添加新信息，以及如何利用它们从隐藏状态中删除不需要的信息？

首先，我们将看到如何向隐藏状态添加新信息。

我们创建一个名为**内容状态**的新状态，![](img/1cf0e78d-66a7-4773-81be-1397536c6f58.png)，用于保存信息。我们知道重置门用于删除不需要的信息。因此，利用重置门，我们创建一个仅包含所需信息的内容状态，![](img/1cf0e78d-66a7-4773-81be-1397536c6f58.png)。

在时间步骤 ![](img/fc68ff5a-bba6-4753-ba76-9515b7ece4e9.png)，内容状态 ![](img/a7eca8be-17cf-401e-a3b2-95d2477d60df.png) 表示如下：

![](img/21e4b03f-9d02-4703-a00b-9675c1e0bcd3.png)

下图显示了如何使用重置门创建内容状态：

![](img/ebea80bb-d016-4b93-9a31-74aac916781e.png)

现在我们将看到如何从隐藏状态中删除信息。

我们了解到更新门 ![](img/e09c8a23-2a52-4a5e-86d1-106944f8980b.png) 帮助确定上一个时间步 ![](img/a6a01139-221f-46f7-88be-ca70346c67cd.png) 中哪些信息可以传递到下一个时间步 ![](img/deb19d44-54c9-444c-b4f8-50a6991a1812.png)。将 ![](img/f8dc0900-cf2a-4754-9c4f-1d24724daece.png) 和 ![](img/85a8fa1c-a7d8-4def-93ba-15c9acbe7a47.png) 相乘，我们仅获取上一步骤中的相关信息。而不是使用新门，我们只是使用 ![](img/d5b1a3a7-ce26-4e57-b714-896e9e27e8c2.png) 的补集，即 ![](img/eb7ba754-ed5d-4b47-a9ad-e15e197602c3.png)，并将其与 ![](img/b3a41778-5ef9-46f7-89f1-2883a5988774.png) 相乘。

随后，隐藏状态更新如下：

![](img/267cbcdf-84e8-4996-a3db-24b225663e6b.png)

一旦计算出隐藏状态，我们可以应用 softmax 函数并计算输出如下：

![](img/96c12b33-2dc6-4377-9cce-b2d173c0d139.png)

# GRU 单元的前向传播

将所有这些内容结合起来，我们在前一节中学到，GRU 单元中完整的前向传播步骤可以表示如下：

+   **更新门**：![](img/847e9d1e-f3ef-4209-b5be-00f938f5ee8c.png)

+   **重置门**：![](img/8a3a5b9f-54f4-4904-9ac0-2c2fb0290d37.png)

+   **内容状态**：![](img/4ee3853c-c054-46d0-86d9-d958683b5d8b.png)

+   **隐藏状态**：![](img/13ad384e-3b6a-431d-a00e-9a3694dee3c6.png)

+   **输出**：![](img/aeab6138-f758-453e-8630-e45f9af2952c.png)

# GRU 单元中的反向传播

总损失，![](img/63ab3607-3992-42fe-b860-6c3304219d4b.png)，是所有时间步骤上损失的总和，可以表示如下：

![](img/0702a3f5-c713-485e-8e8b-7e2e2af1ef7d.png)

为了通过梯度下降最小化损失，我们找出 GRU 单元中所有权重的损失导数如下：

+   我们有三个输入到隐藏层权重，![](img/1a9494ac-cb8d-4d6d-846d-8f456bbf67f6.png)，分别是更新门、重置门和内容状态的输入到隐藏层权重

+   我们有三个隐藏到隐藏层权重，![](img/3dd8d867-6db0-4ad7-bb26-788ae1a0cd90.png)，分别是更新门、重置门和内容状态的隐藏到隐藏层权重

+   我们有一个隐藏到输出层权重，![](img/a9ce24d1-a366-4c22-bda6-400cd788eda4.png)

通过梯度下降找到所有这些权重的最优值，并根据权重更新规则更新权重。

# 门的梯度

正如我们在讨论 LSTM 单元时看到的那样，计算所有权重的损失梯度需要考虑所有门和内容状态的梯度。因此，首先我们将看看如何计算它们。

在接下来的计算中，我们将使用损失相对于隐藏状态的梯度，![](img/923169e7-19ba-4dc0-8a13-4214bbb178ce.png)，在多个地方为![](img/ddef7679-ba54-482d-ab14-333662ddc788.png)，因此我们将看看如何计算它。计算损失相对于隐藏状态的梯度，![](img/4148809c-68e3-4c3e-9706-980b7bd27bfc.png)，与我们在 LSTM 单元中看到的完全相同，可以如下给出：

![](img/f598ff53-6808-4463-804a-f531587fe763.png)

首先，让我们看看如何计算与**内容状态相关的损失梯度**，**![](img/4e5f3d3c-8df9-4ac7-89c3-0d4e8d96b620.png)**。

要计算与内容状态相关的损失梯度，请查看正向传播的方程，并找出哪个方程式有![](img/1928409a-0882-4021-bbe0-f29c523ec6fe.png)项。在隐藏状态方程式中，也就是方程 *(8)* 中，我们有![](img/8bac39e4-c1e7-476c-acef-5cf6db18ec0d.png)项，如下所示：

![](img/5ed2980b-f1d6-4508-91a1-f61928ff475a.png)

因此，根据链式法则，我们可以写成如下形式：

![](img/34b11e23-4329-4076-9916-51b6e40fc15a.png)

![](img/9f527edc-d8dc-4f7f-8a97-4a5e4f1764a6.png)

让我们看看如何计算**重置门![](img/445168d8-588f-4862-b9f5-ce99849bfa1c.png)**的损失梯度。

我们在内容状态方程中有**![](img/445168d8-588f-4862-b9f5-ce99849bfa1c.png)**项，并且可以表示如下：

![](img/e80c58f3-9cfb-4074-a4a7-008fe5411bff.png)

因此，根据链式法则，我们可以写成如下形式：

![](img/ab951795-aa0c-4aab-8e2c-b97201cecbb5.png)

![](img/e628d532-0d57-4b99-81c5-03ed86cb702c.png)

最后，我们看到与更新门相关的**损失梯度**，**![](img/0c368ec1-8c86-411c-bb37-2aec3661376e.png)**。

在我们的隐藏状态方程 ![](img/ef8eadfa-f6cf-4863-8bb6-9725aa8c6486.png) 中，我们有一个项， ![](img/f121ba04-5390-40f5-973c-563a17ad4305.png)，该方程可以表示如下：

![](img/cbc9da3a-b8eb-4856-b11b-9b0cbc696e0c.png)

因此，根据链式法则，我们可以写出以下内容：

![](img/a5402022-2f76-4529-bcb1-0bcc89805103.png)

![](img/fbcf38cd-f3cb-4b0e-9804-c0a252c96bcc.png)

我们已经计算了对所有门和内容状态的损失梯度，现在我们将看看如何计算对我们的 GRU 单元中所有权重的损失梯度。

# 权重的梯度

现在，我们将看到如何计算 GRU 单元中使用的所有权重的梯度。

# 相对于 V 的梯度

由于 GRU 的最终方程，即 ![](img/aa716d72-cf77-4c3a-b868-177796228d2f.png)，与 RNN 相同，计算损失相对于隐藏到输出层权重 ![](img/fd2c30d3-d234-4723-a34f-3ed01e446a93.png) 的梯度与我们在 RNN 中计算的完全相同。因此，我们可以直接写出以下内容：

![](img/d73e3eea-1053-42f3-a11d-2019c45a736b.png)

# 相对于 W 的梯度

现在，我们将看看如何计算对所有门和内容状态中使用的隐藏到隐藏层权重 ![](img/788453d8-602f-4fc5-a548-41f56f1a4211.png) 的损失梯度。

让我们计算**相对于** ![](img/32102421-1ca5-405c-a7e4-b309392b8f93.png) 的损失梯度。

回想一下重置门方程，如下所示：

![](img/97cf5199-4f32-49d8-83f3-ed7693ade454.png)

使用链式法则，我们可以写出以下内容：

![](img/487945b5-2b81-409e-94c0-bddd6b37bbad.png)

让我们计算前述方程中的每个项。第一项， ![](img/fe0f7034-db54-4457-b4ab-f432c716e32e.png)，我们在方程 *(11)* 中已经计算过。第二项计算如下：

![](img/2ffde3db-4f9b-468b-a843-bd464ac69bb1.png)

因此，我们计算 ![](img/f5c2f6ff-bcba-4453-b2ad-efafd43a4c6f.png) 的损失梯度的最终方程如下：

![](img/305bc7f0-e404-4f2d-bee8-3127457c22da.png)

现在，让我们继续找出**相对于** ![](img/77d0b331-8cf2-4007-a59c-564ee2ca0720.png) 的损失梯度。

回想一下更新门方程，如下所示：

![](img/141ee87d-fdcf-49a7-9e5c-cc3566e62fc1.png)

使用链式法则，我们可以写出以下内容：

![](img/6eaff2f8-a5b1-4c20-993d-f8c5876bb000.png)

我们已经计算了方程 *(12)* 中的第一项。第二项计算如下：

![](img/203434df-078f-44be-9968-3cf74849a178.png)

因此，我们计算 ![](img/bd20ff27-1633-40d0-a72b-7ca273217dcf.png) 的损失梯度的最终方程如下：

![](img/fe1aa551-62a2-4e10-bf5f-7dc68b146491.png)

现在，我们将找出**相对于** ![](img/01668886-4be0-44a5-9f6e-15a4a909b677.png) 的损失梯度。

回想一下内容状态方程：

![](img/195a4d7d-2353-4c6c-8c1b-35d8db0a03fd.png)

使用链式法则，我们可以写出如下内容：

![](img/2c34b6db-257e-4dfa-8563-98d508bb5330.png)

参考方程 *(10)* 的第一项。第二项如下所示：

![](img/505f096b-8390-4702-b18a-3e491886e12f.png)

因此，我们计算损失相对于 ![](img/043b0ae1-3783-45ba-91a8-c690cee78bc9.png) 的梯度的最终方程如下：

![](img/f5d54ee1-6c3d-4dc6-bb26-4ee28c5d6240.png)

# 相对于 U 的梯度

现在我们将看到如何计算损失相对于隐藏权重输入 ![](img/0b8de361-8ecb-4102-9c63-75dbdf526d46.png) 的梯度，适用于所有门和内容状态。相对于 ![](img/0b8de361-8ecb-4102-9c63-75dbdf526d46.png) 的梯度与相对于 ![](img/b16fc05c-9ed5-4023-b394-2c14fe1045fd.png) 的计算完全相同，除了最后一项将是 ![](img/01ccd390-2098-4826-bad5-9c2cddc0e47a.png) 而不是 ![](img/982aaf04-59cd-4cd9-a220-6c2d024224b1.png)，这与我们在学习 LSTM 单元时学到的类似。

我们可以将损失相对于 ![](img/31605843-c79f-4a17-9b7c-c53413e19212.png) 的梯度写成：

![](img/adad0428-f7e0-428c-a107-b09bdef4e19c.png)

**损失相对于** ![](img/8c284527-0e48-4887-b8f2-918f353a812e.png) 的梯度表示如下：

![](img/85f1ffc8-bf99-4503-bf02-210c5a205d83.png)

**损失相对于** ![](img/ec18e7d7-2132-4be0-95c9-9de487839d73.png) 的梯度表示如下：

![](img/1516e03a-d4c4-427c-9c78-aa6e160553b1.png)

# 在 TensorFlow 中实现 GRU 单元

现在，我们将看到如何在 TensorFlow 中实现 GRU 单元。而不是查看代码，我们只会看到如何在 TensorFlow 中实现 GRU 的前向传播。

# 定义权重

首先，让我们定义所有权重。更新门的权重定义如下：

```py
 Uz = tf.get_variable("Uz", [vocab_size, hidden_size], initializer=init)
 Wz = tf.get_variable("Wz", [hidden_size, hidden_size], initializer=init)
 bz = tf.get_variable("bz", [hidden_size], initializer=init)
```

重置门的权重如下定义：

```py
Ur = tf.get_variable("Ur", [vocab_size, hidden_size], initializer=init)
Wr = tf.get_variable("Wr", [hidden_size, hidden_size], initializer=init)
br = tf.get_variable("br", [hidden_size], initializer=init)
```

内容状态的权重定义如下：

```py
Uc = tf.get_variable("Uc", [vocab_size, hidden_size], initializer=init)
Wc = tf.get_variable("Wc", [hidden_size, hidden_size], initializer=init)
bc = tf.get_variable("bc", [hidden_size], initializer=init)
```

输出层的权重定义如下：

```py
V = tf.get_variable("V", [hidden_size, vocab_size], initializer=init)
by = tf.get_variable("by", [vocab_size], initializer=init)
```

# 定义前向传播

将更新门定义为方程 *(5)* 中所给定的：

```py
zt = tf.sigmoid(tf.matmul(x_t, Uz) + tf.matmul(h_t, Wz) + bz)
```

将重置门定义为方程 *(6)* 中所给定的：

```py
rt = tf.sigmoid(tf.matmul(x_t, Ur) + tf.matmul(h_t, Wr) + br)
```

将内容状态定义为方程 *(7)* 中所给定的：

```py
ct = tf.tanh(tf.matmul(x_t, Uc) + tf.matmul(tf.multiply(rt, h_t), Wc) + bc)
```

将隐藏状态定义为方程 *(8)* 中所给定的：

```py
 h_t = tf.multiply((1 - zt), ct) + tf.multiply(zt, h_t)
```

根据方程 *(9)* 计算输出：

```py
 y_hat_t = tf.matmul(h_t, V) + by
```

# 双向 RNN

在双向 RNN 中，我们有两个不同的隐藏单元层。这两层从输入层到输出层连接。在一层中，隐藏状态从左到右共享，在另一层中，它们从右到左共享。

但这意味着什么？简单地说，一个隐藏层从序列的起始点向前移动通过时间，而另一个隐藏层从序列的末尾向后移动通过时间。

如下图所示，我们有两个隐藏层：前向隐藏层和后向隐藏层，具体描述如下：

+   在前向隐藏层中，隐藏状态值是从过去的时间步共享的，即，![](img/b085442d-8fcf-432e-b043-965a97c0568a.png) 被共享到![](img/a81ec639-6e9f-42e4-bee2-3f2254f8af4c.png)，![](img/61ae5015-89e5-48c7-a39f-24721e059c4a.png) 被共享到![](img/b574afd9-1cd4-4654-b3b5-6564485798e0.png)，依此类推。

+   在后向隐藏层中，隐藏起始值是从未来的时间步共享的，即，![](img/992a0d8a-6e5a-471b-99e6-c6a7e56d65a3.png)到![](img/d02c4d9a-11d6-4231-adb9-12c324054f7e.png)，![](img/6e62c00e-9b75-4e01-b4c8-369bb83bcd8e.png)到![](img/48cdf1b9-46a3-4eb6-99e9-48200baa02d7.png)，依此类推。

前向隐藏层和后向隐藏层如下图所示：

![](img/2496305a-f7ff-42dd-9e6d-ff080eaff0b7.png)

双向循环神经网络的用途是什么？在某些情况下，从两个方向读取输入序列非常有用。因此，双向循环神经网络包括两个 RNN，一个从前向读取句子，另一个从后向读取句子。

例如，考虑以下句子：

*阿奇在 _____ 待了 13 年。所以他擅长说中文。*

如果我们使用 RNN 来预测上述句子中的空白，这将是模棱两可的。正如我们所知，RNN 只能根据它迄今为止看到的一组词来进行预测。在上述句子中，要预测空白，RNN 只看到*阿奇*、*待了*、*13*、*年*、*在*，*但*这些词汇单独并不提供足够的上下文，也不能清楚地预测出正确的单词。它只是说*阿奇在待了 13 年在.*仅仅凭这些信息，我们无法正确预测接下来的单词。

但是，如果我们也阅读空白后面的单词，如*所以，他，是，擅长，说，中文*和*中国*，那么我们可以说阿奇在*中国*待了 13 年，因为他擅长说中文。因此，在这种情况下，如果我们使用双向循环神经网络来预测空白，它将能够正确预测，因为它在做出预测之前会同时从前向和后向读取句子。

双向循环神经网络已经在各种应用中使用，例如**词性标注（POS）**，在这种情况下，知道目标词前后的词汇是至关重要的，语言翻译，预测蛋白质结构，依赖句法分析等。然而，双向循环神经网络在不知道未来信息的在线设置中不适用。

双向循环神经网络的前向传播步骤如下所示：

+   前向隐藏层：

![](img/0d43e50d-cad2-41ee-b068-45c881e13b49.png)

+   后向隐藏层：

![](img/050f71ab-72fc-408e-88b2-e10df4889c4a.png)

+   输出：

![](img/4bb182a9-6c94-4ad0-b041-54002149ad6b.png)

使用 TensorFlow 实现双向递归神经网络很简单。假设我们在双向递归神经网络中使用 LSTM 单元，我们可以按以下步骤操作：

1.  从 TensorFlow 的 `contrib` 中导入 `rnn`，如下所示：

```py
from tensorflow.contrib import rnn
```

1.  定义前向和后向隐藏层：

```py
forward_hidden_layer = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

backward_hidden_layer = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
```

1.  使用`rnn.static_bidirectional_rnn`来定义双向递归神经网络：

```py
outputs, forward_states, backward_states = rnn.static_bidirectional_rnn(forward_hidden_layer, backward_hidden_layer, input)                                         
```

# 深入理解深度递归神经网络（deep RNN）

我们知道，深度神经网络是具有多个隐藏层的网络。类似地，深度递归神经网络具有多个隐藏层，但当我们有多个隐藏层时，隐藏状态如何计算呢？我们知道，递归神经网络通过接收输入和先前的隐藏状态来计算隐藏状态，但当我们有多个隐藏层时，后续层的隐藏状态如何计算呢？

例如，让我们看看隐藏层 2 中的 ![](img/32e70998-4576-4af1-aa2b-6b05010aafdd.png) 如何计算。它接收前一个隐藏状态 ![](img/1eb977fd-947d-4b96-85aa-df56ef1a212e.png) 和前一层的输出 ![](img/6da43a1a-a146-483a-a5a7-6ed2e7af391a.png) 作为输入来计算 ![](img/b125c9ce-dca1-4210-b87b-a64dc4cf9626.png)。

因此，当我们有多个隐藏层的递归神经网络时，后续层的隐藏层将通过接收前一个隐藏状态和前一层的输出作为输入来计算，如下图所示：

![](img/f7a7594f-1f84-45c1-9691-c781cad36333.png)

# 使用 seq2seq 模型进行语言翻译

**序列到序列模型**（**seq2seq**）基本上是 RNN 的一对多架构。它已被用于各种应用，因为它能够将任意长度的输入序列映射到任意长度的输出序列。seq2seq 模型的一些应用包括语言翻译、音乐生成、语音生成和聊天机器人。

在大多数实际场景中，输入和输出序列的长度是变化的。例如，让我们考虑语言翻译任务，在这个任务中，我们需要将一种语言的句子转换为另一种语言。假设我们将英语（源语言）转换为法语（目标语言）。

假设我们的输入句子是*what are you doing?*，那么它将被映射为*que faites vous?* 如我们所见，输入序列由四个单词组成，而输出序列由三个单词组成。seq2seq 模型可以处理这种不同长度的输入和输出序列，并将源序列映射到目标序列。因此，在输入和输出序列长度变化的应用中广泛使用它们。

seq2seq 模型的架构非常简单。它包括两个关键组件，即编码器和解码器。让我们考虑同样的语言翻译任务。首先，我们将输入句子馈送给编码器。

编码器学习输入句子的表示（嵌入），但是什么是表示？表示或嵌入基本上是包含句子意义的向量。它也被称为**思想向量**或**上下文向量**。一旦编码器学习了嵌入，它将嵌入发送到解码器。解码器将这个嵌入（思想向量）作为输入，并试图构造目标句子。因此，解码器尝试为英语句子生成法语翻译。

如下图所示，编码器接收输入的英语句子，学习嵌入，并将嵌入馈送给解码器，然后解码器使用这些嵌入生成翻译后的法语句子：

![](img/cb406d2e-a21b-4815-894b-6eb66da3ad8a.png)

但这是如何真正工作的呢？编码器如何理解句子？解码器如何使用编码器的嵌入翻译句子？让我们深入探讨一下，看看这是如何运作的。

# 编码器

编码器基本上是带有 LSTM 或 GRU 单元的 RNN。它也可以是双向 RNN。我们将输入句子馈送给编码器，而不是获取输出，而是从最后一个时间步获取隐藏状态作为嵌入。让我们通过一个示例更好地理解编码器。

考虑我们使用的是带有 GRU 单元的 RNN，输入句子是*what are you doing.* 让我们用*e*表示编码器的隐藏状态：

![](img/4c41e25d-29b2-4797-bb1b-c55daf55f12c.png)

前面的图示展示了编码器如何计算思想向量；下文将详细解释：

+   在第一个时间步中，我们传递输入，![](img/e34b07fb-7a00-433c-8db9-7dc532d394fd.png) 给一个 GRU 单元，这是输入句子中的第一个词*what*，以及初始隐藏状态，![](img/5e6d805a-5f4c-4be0-9c1d-aea4cb747b6d.png)，它是随机初始化的。使用这些输入，GRU 单元计算第一个隐藏状态，![](img/17df6ef9-4256-4fd0-8adf-31e46c59cae5.png)，如下所示：

![](img/e418bc62-3d67-40a4-9bdb-f1ca15a275e9.png)

+   在下一个时间步中，我们传递输入，![](img/3fbeb7b9-f455-4f83-a654-3d77aa45f0cf.png)，这是输入句子中的下一个词*are*，给编码器。除此之外，我们还传递上一个隐藏状态，![](img/b1ceff9f-3617-4ec6-9885-c96bfe885a10.png)，并计算隐藏状态，![](img/d8dafef5-2c21-4bba-a469-64762a296725.png)：

![](img/ffaad91e-ab69-429d-bdbc-cdbd3b7b4eba.png)

+   在下一个时间步中，我们传递输入，![](img/6de00d57-0eb1-454c-a7f4-6129e6e8a89f.png)，这是下一个词*you*，给编码器。除此之外，我们还传递上一个隐藏状态，![](img/316bbbe8-dfed-4391-9a1d-f2ff08a6e10a.png)，并计算隐藏状态，![](img/9c195ab0-4ec6-4b54-b884-c0e94c56ad32.png) 如下所示：

![](img/fed1de64-bb41-4794-90ea-487785fb7cd6.png)

+   在最后一个时间步 ![](img/1fb887d2-d3c9-41ed-b750-0a9891e4f39b.png) 中，我们输入 *doing.* 作为输入单词。同时传递前一个隐藏状态 ![](img/6b36f190-3001-4333-b5ff-92b7aee41ce9.png)，计算隐藏状态 ![](img/563552de-06a9-4f23-bd52-5f6aa84daccf.png)：

![](img/a739f402-78c5-4264-b9aa-3ad155084ec0.png)

因此，![](img/ed565eaf-65a4-4944-9907-c5cc0e9003d9.png) 是我们的最终隐藏状态。我们了解到 RNN 在其隐藏状态中捕捉到目前为止看到的所有单词的上下文。由于 ![](img/ed565eaf-65a4-4944-9907-c5cc0e9003d9.png) 是最终隐藏状态，它包含了网络看到的所有单词的上下文，即我们输入句子中的所有单词，即 *what, are, you* 和 *doing.* 

由于最终隐藏状态 ![](img/ed565eaf-65a4-4944-9907-c5cc0e9003d9.png) 包含了输入句子中所有单词的上下文，因此它包含了输入句子的上下文，并且这实质上形成了我们的嵌入 ![](img/f16c2c9b-4d5a-4aff-a250-081783a4b22f.png)，也称为思考向量或上下文向量，如下所示：

![](img/64336f60-c1ee-43f3-b18a-e53d79fc13a4.png)

我们将上下文向量 ![](img/873ee16b-63e7-4b6e-84f5-ab8d4da349ef.png) 传递给解码器，以将其转换为目标句子。

因此，在编码器中，每个时间步 ![](img/cb86a3c5-8094-481c-b6a3-aeba3898ba33.png)，我们输入一个单词，并与之前的隐藏状态 ![](img/2e811f4c-cafc-481a-bb02-e4245d8ce3f6.png) 一起计算当前的隐藏状态 ![](img/f87b5d21-8d53-45c5-b560-f9d0e5a935c7.png)。最终步骤中的隐藏状态 ![](img/21ba6237-3c97-44c7-aed4-5c311a216198.png) 包含了输入句子的上下文，并将成为嵌入 ![](img/86d73894-d18d-4d2c-947a-dfa489832953.png)，该嵌入将发送到解码器以将其转换为目标句子。

# 解码器

现在，我们将学习如何使用编码器生成的思考向量 ![](img/5611c453-0f79-42da-badf-e2f0d9457e5c.png) 来生成目标句子。解码器是一个带有 LSTM 或 GRU 单元的 RNN。我们的解码器的目标是为给定的输入（源）句子生成目标句子。

我们知道，我们通过使用随机值初始化 RNN 的初始隐藏状态来启动它，但对于解码器的 RNN，我们初始化隐藏状态的方式是使用由编码器生成的思考向量，![](img/5611c453-0f79-42da-badf-e2f0d9457e5c.png)，而不是使用随机值。解码器网络如下图所示：

![](img/16201a52-10b3-4e44-99bb-2360b2e765fd.png)

但是，解码器的输入应该是什么？我们简单地将**<sos>**作为解码器的输入，表示句子的开始。因此，一旦解码器收到**<sos>**，它尝试预测目标句子的实际起始词。让我们用![](img/f55e47fa-02bb-4d56-9cee-6ae073459bc9.png)表示解码器的隐藏状态。

在第一个时间步骤，![](img/bb6f3248-d10c-430b-904b-3fd6b8dcfbf3.png)，我们将第一个输入**<sos>**传递给解码器，并且还传递思考向量作为初始隐藏状态，如下所示：

![](img/c3af7b26-63b2-4d53-baf1-fb4af71207ff.png)

好的。我们到底在做什么？我们需要预测输出序列，即我们输入的英语句子的法语等价物。我们的词汇表中有很多法语单词。解码器如何决定输出哪个单词？也就是说，它如何确定输出序列的第一个单词？

我们将解码器隐藏状态![](img/17b762eb-33fd-4586-ac58-8f924f094aaa.png)馈送到![](img/aee8660c-9dda-48a3-be0a-0a61af8bc3e7.png)，它返回所有词汇表中的分数，作为第一个输出词。也就是说，在时间步骤![](img/a500d9e6-c50f-458d-addb-e9763adb785c.png)的输出词计算如下：

![](img/54b3e377-4c58-46c9-b529-e08a64d678b7.png)

我们不是直接使用原始分数，而是将它们转换为概率。由于我们了解到 softmax 函数将值压缩到 0 到 1 之间，我们使用 softmax 函数将分数![](img/25c8b470-88c7-4d5e-9272-f6eee9c2046e.png)转换为概率![](img/99b1c5e2-d4a4-41e0-aee5-97e51f839191.png)：

![](img/f232af41-0339-40c2-b870-a0a68db6d0f3.png)

因此，我们得到了所有法语词汇中第一个输出词的概率。我们使用 argmax 函数选择具有最高概率的词作为第一个输出词：

![](img/5e427382-3160-4167-928a-c4994658f405.png)

因此，我们预测第一个输出词![](img/f0186d5c-499b-4870-b087-725c99e997d5.png)为*Que*，如前图所示。

在下一个时间步骤![](img/d37e3c9e-9a88-4c20-a355-9addddfcb9e4.png)，我们将前一个时间步骤预测的输出词![](img/af587c78-f7e6-4aab-b8e6-0050d6daf143.png)作为解码器的输入。同时，我们还传递上一个隐藏状态![](img/fc6b17ac-7414-4abb-8bed-bb1765234945.png)：

![](img/3d8355c8-f013-42a2-8805-3fcd8d358ddb.png)

接着，我们计算所有词汇表中的分数，作为下一个输出词，即时间步骤![](img/3629c1fb-1a2f-44fb-93b1-41e845b698e0.png)的输出词：

![](img/e8831d28-a8bc-467c-a577-dbe515b496da.png)

然后，我们使用 softmax 函数将分数转换为概率：

![](img/ca4ae24d-bb32-4d87-959f-76299a956420.png)

接下来，我们选择具有最高概率的单词作为输出词，![](img/5d0f3231-b41f-44c0-b2a0-7a814bf59961.png)，在时间步骤![](img/c79af8e1-8071-494a-8035-5dbd83febde7.png)：

![](img/0b2cb5ad-7c33-43b3-a6c7-c186a330cb0f.png)

因此，我们使用 ![](img/d4ee253b-da1a-4e03-a597-fd7962fbef9d.png) 初始化解码器的初始隐藏状态，并且在每个时间步 ![](img/a6175e48-f082-41d3-b5a0-8d88cd81acdd.png) 中，我们将来自上一个时间步的预测输出词 ![](img/0d3ea9bc-9fd3-4388-bc27-0dd6ceb7da97.png) 和先前的隐藏状态 ![](img/46f55371-9ab0-4dff-aacb-8afb1de0a904.png) 作为解码器当前时间步骤的输入 ![](img/6abe5d13-c61e-4ae8-9ead-a97dbaf01f25.png)，并预测当前输出 ![](img/9460279a-8946-4b1b-a9dd-94c2a6618c1f.png)。

但是解码器何时停止？因为我们的输出序列必须在某处停止，我们不能不断将前一个时间步的预测输出词作为下一个时间步的输入。当解码器预测输出词为 **<sos>** 时，这意味着句子的结束。然后，解码器学习到输入源句子被转换为一个有意义的目标句子，并停止预测下一个词。

因此，这就是 seq2seq 模型如何将源句子转换为目标句子。

# 注意力就是我们所需的一切

我们刚刚学习了 seq2seq 模型的工作原理以及它如何将源语言的句子翻译成目标语言的句子。我们了解到上下文向量基本上是来自编码器最终时间步的隐藏状态向量，它捕捉了输入句子的含义，并由解码器用于生成目标句子。

但是当输入句子很长时，上下文向量不能捕捉整个句子的含义，因为它只是来自最终时间步的隐藏状态。因此，我们不再将最后一个隐藏状态作为上下文向量并用于解码器，而是取编码器所有隐藏状态的总和作为上下文向量。

假设输入句子有 10 个单词；那么我们将有 10 个隐藏状态。我们将所有这些 10 个隐藏状态求和，并将其用于解码器生成目标句子。然而，并非所有这些隐藏状态在生成时间步骤 ![](img/6936ee89-bfce-44d9-b004-e6c5c05492d4.png) 时都可能有帮助。有些隐藏状态比其他隐藏状态更有用。因此，我们需要知道在时间步骤 ![](img/6936ee89-bfce-44d9-b004-e6c5c05492d4.png) 时哪个隐藏状态比另一个更重要来预测目标词。为了获得这种重要性，我们使用注意力机制，它告诉我们在时间步骤 ![](img/6936ee89-bfce-44d9-b004-e6c5c05492d4.png) 时哪个隐藏状态更重要以生成目标词。因此，注意力机制基本上为编码器的每个隐藏状态在时间步骤 ![](img/6936ee89-bfce-44d9-b004-e6c5c05492d4.png) 生成目标词提供重要性。

注意力机制如何工作？假设我们有编码器的三个隐藏状态 ![](img/ab2fa4ea-aaff-4377-bc9e-a0b28296284c.png)、![](img/5860915c-794c-4fdd-ad57-8e642566f4f6.png) 和 ![](img/30024c5d-abdb-4ce5-9cdc-71871912ce1e.png)，以及解码器的隐藏状态 ![](img/9368e002-9c63-44b1-b2d0-99f41fb3dfbc.png)，如下图所示：

![](img/50f65fd0-d5ba-4da2-a699-b23638e790e5.png)

现在，我们需要了解编码器所有隐藏状态在时间步 ![](img/6936ee89-bfce-44d9-b004-e6c5c05492d4.png) 生成目标词的重要性。因此，我们取每个编码器隐藏状态 ![](img/e6f7c741-6fb6-4bcd-b238-37a217a333bf.png) 和解码器隐藏状态 ![](img/88105601-2aa2-47b8-8746-306e23c3ee0c.png)，并将它们输入到一个称为**分数函数**或**对齐函数**的函数 ![](img/a835122d-5352-40d5-93ef-213096cfc65c.png) 中，它返回每个编码器隐藏状态的分数，指示它们的重要性。但这个分数函数是什么？分数函数有多种选择，如点积、缩放点积、余弦相似度等。

我们使用简单的点积作为分数函数；即编码器隐藏状态和解码器隐藏状态之间的点积。例如，要了解生成目标词 ![](img/7c724e95-9b8e-4f08-bf46-c5548b025209.png) 的重要性，我们简单地计算 ![](img/7c724e95-9b8e-4f08-bf46-c5548b025209.png) 和 ![](img/7b552100-35cf-4ec3-bd70-d4c26226c57c.png) 之间的点积，这给我们一个指示 ![](img/7c724e95-9b8e-4f08-bf46-c5548b025209.png) 和 ![](img/7b552100-35cf-4ec3-bd70-d4c26226c57c.png) 相似程度的分数。

一旦我们得到分数，我们将它们使用 softmax 函数转换为概率，如下所示：

![](img/444b78ec-2865-4896-ada8-4d8d2ff24499.png)

这些概率 ![](img/981305de-f7b0-4f08-a314-5ddf7dcb8cec.png) 被称为**注意力权重**。

如下图所示，我们计算每个编码器隐藏状态与解码器隐藏状态之间的相似性分数，使用一个函数 ![](img/b584b42e-3bde-4b0a-9fc0-92a6b9044445.png)。然后，使用 softmax 函数将相似性分数转换为概率，称为注意力权重：

![](img/a795cfb5-5829-4874-b642-9b8c64624ab4.png)

因此，我们得到每个编码器隐藏状态的注意力权重（概率）。现在，我们将注意力权重乘以它们对应的编码器隐藏状态，即 ![](img/a24e9bf0-819d-41ef-9958-0556abd202fa.png)。如下图所示，编码器的隐藏状态 ![](img/842aced5-251e-47ac-84fe-fe4ba8dacc32.png) 乘以 **0.106**，![](img/b225d1d3-8668-4509-b83f-890f57a03688.png) 乘以 **0.106**，![](img/124f9743-e3df-43c9-ab5b-5d6780b61fa4.png) 乘以 **0.786**：

![](img/d4b71126-368c-47cb-bbb4-d8c3b525a8ee.png)

但是，为什么我们要将注意力权重乘以编码器的隐藏状态？

将编码器的隐藏状态乘以它们的注意力权重表示我们正在赋予那些具有更多注意力权重的隐藏状态更重要的重视，而对具有较少注意力权重的隐藏状态则不那么重视。如前图所示，将**0.786**乘以隐藏状态![](img/124f9743-e3df-43c9-ab5b-5d6780b61fa4.png)意味着我们比其他两个隐藏状态更重视![](img/124f9743-e3df-43c9-ab5b-5d6780b61fa4.png)。

因此，这就是注意机制如何决定哪个隐藏状态在时间步骤![](img/99f9244a-a159-4c7a-8a0e-2bb3bded1610.png)生成目标词。在将编码器的隐藏状态乘以它们的注意力权重后，我们简单地将它们相加，这现在形成我们的上下文/思想向量：

![](img/0b35b4ea-9e72-442b-85a3-1d67a98b7f62.png)

如下图所示，上下文向量是通过将编码器的隐藏状态乘以其相应的注意力权重后得到的总和：

![](img/ff2ac115-5a3d-4f35-927b-ded90f66b89f.png)

因此，为了在时间步骤![](img/99f9244a-a159-4c7a-8a0e-2bb3bded1610.png)生成目标词，解码器使用时间步骤![](img/df8ae413-f3a8-4d68-b6c6-6495a04084f3.png)的上下文向量![](img/a3f119f5-ae5c-485d-af13-98f8387e724c.png)。通过注意机制，我们不再将最后一个隐藏状态作为上下文向量并用于解码器，而是取所有编码器隐藏状态的总和作为上下文向量。

# 总结

在本章中，我们学习了 LSTM 单元如何使用多个门来解决梯度消失问题。然后，我们看到如何在 TensorFlow 中使用 LSTM 单元来预测比特币的价格。

在查看了 LSTM 单元之后，我们了解了 GRU 单元，它是 LSTM 的简化版本。我们还学习了双向 RNN，其中我们有两层隐藏状态，一层从序列的起始时间向前移动，另一层从序列的末尾时间向后移动。

在本章末尾，我们了解了 seq2seq 模型，它将一个长度不同的输入序列映射到一个长度不同的输出序列。我们还了解了注意机制如何在 seq2seq 模型中使用，以及它如何集中关注重要信息。

在下一章中，我们将学习卷积神经网络及其在识别图像中的应用。

# 问题

让我们将新学到的知识付诸实践。回答以下问题：

1.  LSTM 如何解决 RNN 的梯度消失问题？

1.  LSTM 单元中所有不同门及其功能是什么？

1.  细胞状态的用途是什么？

1.  GRU 是什么？

1.  双向 RNN 如何工作？

1.  深层 RNN 如何计算隐藏状态？

1.  在 seq2seq 架构中，编码器和解码器是什么？

1.  注意机制有什么用途？

# 进一步阅读

在 GitHub 上查看一些很酷的项目：

+   使用 LSTM 进行人体活动识别：[`github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition`](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)

+   使用 seq2seq 构建聊天机器人：[`github.com/tensorlayer/seq2seq-chatbot`](https://github.com/tensorlayer/seq2seq-chatbot)

+   使用双向 GRU 进行文本摘要：[`github.com/harpribot/deep-summarization`](https://github.com/harpribot/deep-summarization)
