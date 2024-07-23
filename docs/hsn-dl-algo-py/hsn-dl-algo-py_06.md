# 使用 RNN 生成歌词

在普通前馈神经网络中，每个输入都是独立的。但是对于序列数据集，我们需要知道过去的输入以进行预测。序列是一组有序的项目。例如，一个句子是一个单词序列。假设我们想要预测句子中的下一个单词；为此，我们需要记住之前的单词。普通的前馈神经网络无法预测出正确的下一个单词，因为它不会记住句子的前面单词。在这种需要记住先前输入的情况下（在这种情况下，我们需要记住先前输入以进行预测），为了进行预测，我们使用**递归神经网络**（**RNNs**）。

在本章中，我们将描述如何使用 RNN 对序列数据集进行建模以及如何记住先前的输入。我们将首先研究 RNN 与前馈神经网络的区别。然后，我们将检查 RNN 中的前向传播是如何工作的。

继续，我们将研究**时域反向传播**（**BPTT**）算法，该算法用于训练 RNN。随后，我们将讨论梯度消失和爆炸问题，在训练递归网络时会出现这些问题。您还将学习如何使用 TensorFlow 中的 RNN 生成歌词。

在本章末尾，我们将研究不同类型的 RNN 架构，以及它们在各种应用中的使用。

在本章中，我们将学习以下主题：

+   递归神经网络

+   RNN 中的前向传播

+   时域反向传播

+   梯度消失和爆炸问题

+   使用 RNN 生成歌词

+   不同类型的 RNN 架构

# 引入 RNNs

*太阳在 ____ 中升起。*

如果我们被要求预测前面句子中的空白处，我们可能会说东方。为什么我们会预测东方是正确的词汇在这里？因为我们读了整个句子，理解了上下文，并预测东方是一个适合完成句子的合适词汇。

如果我们使用前馈神经网络来预测空白，它将不能预测出正确的单词。这是因为在前馈网络中，每个输入都是独立的，并且它们仅基于当前输入进行预测，它们不记住前面的输入。

因此，网络的输入将仅仅是前一个空白处的单词，这个单词是 *the*。仅凭这个单词作为输入，我们的网络无法预测出正确的单词，因为它不知道句子的上下文，也就是说它不知道前面一组单词来理解句子的语境并预测合适的下一个单词。

这就是我们使用 RNN 的地方。它们不仅基于当前输入预测输出，还基于先前的隐藏状态。为什么它们必须基于当前输入和先前的隐藏状态来预测输出呢？为什么不能只使用当前输入和先前的输入呢？

这是因为前一个输入仅存储有关前一个单词的信息，而前一个隐藏状态将捕捉到网络迄今所见的句子中所有单词的上下文信息。基本上，前一个隐藏状态 acts like a memory，并捕捉句子的上下文。有了这个上下文和当前输入，我们可以预测相关的单词。

例如，让我们拿同样的句子 *The sun rises in the ____.* 举例。如下图所示，我们首先将单词 *the* 作为输入传递，然后将下一个单词 *sun* 作为输入；但与此同时，我们也传递了上一个隐藏状态，![](img/458c6f8a-d40d-45d3-8198-eaf4aafa46ac.png)。因此，每次传递输入单词时，我们也会传递前一个隐藏状态作为输入。

在最后一步，我们传递单词 *the*，同时传递前一个隐藏状态 ![](img/c8294095-6268-44e5-873a-4e6c0e90f390.png)，它捕捉到了网络迄今为止所见的单词序列的上下文信息。因此，![](img/bf6446ea-a525-4fb6-a28e-593405664893.png) 充当了记忆，存储了网络已经看到的所有先前单词的信息。有了 ![](img/bd898219-78d6-4fd4-a8f3-7dfa95903ae9.png) 和当前输入单词 (*the*)，我们可以预测出相关的下一个单词：

![](img/a2e18a6b-2792-47f8-97c9-863377ab4bde.png)

简言之，RNN 使用前一个隐藏状态作为记忆，捕捉并存储网络迄今所见的上下文信息（输入）。

RNN 广泛应用于涉及序列数据的用例，如时间序列、文本、音频、语音、视频、天气等等。它们在各种自然语言处理（**NLP**）任务中被广泛使用，如语言翻译、情感分析、文本生成等。

# 前馈网络和 RNN 之间的区别

RNN 和前馈网络的比较如下图所示：

![](img/4327bdb1-a3c4-4763-8a6d-82fb97682747.png)

正如您可以在前面的图表中观察到的那样，RNN 在隐藏层中包含一个循环连接，这意味着我们使用前一个隐藏状态与输入一起来预测输出。

仍然感到困惑吗？让我们看看下面展开的一个 RNN 版本。但等等，什么是 RNN 的展开版本？

这意味着我们展开网络以完成一个完整的序列。假设我们有一个包含 ![](img/04c2ff2f-17e8-47b5-9943-3ef499dcd39b.png) 个单词的输入句子；那么我们将有 ![](img/ccf0389e-8454-44c0-a831-7a712a1393dc.png) 到 ![](img/dc35c22d-bba3-4a16-b0f2-aa6ffa5857e9.png) 层，每层对应一个单词，如下图所示：

![](img/d2509d60-cf99-4d3c-9cc4-b368a9e9c212.png)

如您在前图中所见，在时间步 ![](img/8872a7d9-968c-41fe-9887-abbc14a24781.png)，基于当前输入 ![](img/c0b5e079-f0e9-4776-9802-69bc26929f72.png) 和先前的隐藏状态 ![](img/ba821958-371b-41b1-ad62-dfefef803180.png) 预测输出 ![](img/c188d6dd-d163-4906-89d7-c61a2bfdab8f.png)。同样，在时间步 ![](img/3872e1e6-85c5-4424-9683-cd204d2cb891.png)，基于当前输入 ![](img/d6655da4-2f28-463f-9317-7c187a62e859.png) 和先前的隐藏状态 ![](img/aeac31f3-91bd-487d-88fb-f61d87394fa4.png)，预测 ![](img/2bf40a00-a797-4927-8fab-58170517d363.png)。这就是 RNN 的工作原理；它利用当前输入和先前的隐藏状态来预测输出。

# 循环神经网络的前向传播

让我们看看循环神经网络如何使用前向传播来预测输出；但在我们深入探讨之前，让我们熟悉一下符号：

![](img/67aaa29a-50cc-49fe-a893-ee7d2a1d4872.png)

前述图示明了以下：

+   ![](img/e8c14ce3-b877-46b6-9517-401decad5e39.png) 表示输入到隐藏层的权重矩阵

+   ![](img/b538e5f2-eede-4658-85af-8970a3516600.png) 表示隐藏到隐藏层的权重矩阵

+   ![](img/a65d52e4-e147-484f-9fa3-eb6279443766.png) 表示隐藏到输出层的权重矩阵

在时间步 ![](img/cb45a853-41f7-484f-9446-d6f3f23bd3dd.png)，隐藏状态 ![](img/b9242358-ecb8-4726-8326-d92c4a311927.png) 可以计算如下：

![](img/5b43f70d-e24c-4d7f-b483-59754144e72d.png)

也就是说，*时间步 t 的隐藏状态 = tanh([输入到隐藏层权重 x 输入] + [隐藏到隐藏层权重 x 先前隐藏状态])*。

在时间步 ![](img/c16541d9-f156-4c48-83e3-6dc4f35070cc.png)，输出可以如下计算：

![](img/10b8680f-7613-4b12-b473-4a471403db20.png)

也就是说，*时间步 t 的输出 = softmax(隐藏到输出层权重 x 时间步 t 的隐藏状态)*。

我们还可以如下图所示表示循环神经网络（RNN）。正如您所看到的，隐藏层由一个 RNN 块表示，这意味着我们的网络是一个 RNN，并且先前的隐藏状态用于预测输出：

![](img/4ed16907-a459-43dc-9d6c-640e671139a6.png)

下图显示了 RNN 展开版本中前向传播的工作原理：

![](img/edec3143-35b0-4af7-ba04-b37e14030d2a.png)

我们用随机值初始化初始隐藏状态 ![](img/6615f685-b929-45b2-b7e0-e45c559381db.png)。如前图所示，输出 ![](img/50ebe989-e25d-4826-9c9b-71bba2cf8ffa.png) 基于当前输入 ![](img/c32660b6-140a-49df-b3bd-f71b2c1c8987.png) 和先前的隐藏状态（即初始隐藏状态） ![](img/6615f685-b929-45b2-b7e0-e45c559381db.png) 使用以下公式预测：

![](img/1f792fc7-d64f-4a83-902e-2c420fa51d2b.png)

![](img/cd00e1ac-010a-4980-8ca5-2aeab71ccf56.png)

类似地，看看输出 ![](img/f553c432-9a57-444f-97e8-010ee27f60bc.png) 是如何计算的。它使用当前输入 ![](img/f64094cd-9f12-40dd-8916-7d9ecc16810b.png) 和先前的隐藏状态 ![](img/33d251b6-7355-4bdb-8fac-2dc7c7a0d7e3.png)：

![](img/235ea93c-2679-453e-b39c-a69e877df36a.png)

![](img/4e7da6bf-ee7a-4508-a685-3deca9ae7691.png)

因此，在前向传播中，为了预测输出，RNN 使用当前输入和先前的隐藏状态。

为了明确起见，让我们看看如何在 RNN 中实现前向传播以预测输出：

1.  通过从均匀分布中随机抽取，初始化所有权重 ![](img/c3efe389-783a-49bc-836e-476412e21b8a.png)，![](img/0e801a23-d593-4670-866b-8fd1450bf481.png)，和 ![](img/bc9a798b-7e2f-4aff-a0e1-065b94718831.png)：

```py
U = np.random.uniform(-np.sqrt(1.0 / input_dim), np.sqrt(1.0 / input_dim), (hidden_dim, input_dim))

W = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (hidden_dim, hidden_dim))

V = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (input_dim, hidden_dim))
```

1.  定义时间步长的数量，这将是我们输入序列的长度 ![](img/e3ea8ad6-f6b8-42c1-b3df-aedd6220ef0d.png)：

```py
num_time_steps = len(x)
```

1.  定义隐藏状态：

```py
hidden_state = np.zeros((num_time_steps + 1, hidden_dim))
```

1.  用零初始化初始隐藏状态 ![](img/749e77de-caab-41e5-bde8-52a5ad068ecb.png)：

```py
hidden_state[-1] = np.zeros(hidden_dim)
```

1.  初始化输出：

```py
YHat = np.zeros((num_time_steps, output_dim))
```

1.  对于每个时间步长，我们执行以下操作：

```py
for t in np.arange(num_time_steps):

    #h_t = tanh(UX + Wh_{t-1})
    hidden_state[t] = np.tanh(U[:, x[t]] + W.dot(hidden_state[t - 1]))

    # yhat_t = softmax(vh)
    YHat[t] = softmax(V.dot(hidden_state[t]))
```

# 通过时间反向传播

我们刚刚学习了 RNN 中的前向传播如何工作以及如何预测输出。现在，我们计算损失 ![](img/64479a6c-b690-4efc-86a7-517c7dd06b79.png)，在每个时间步 ![](img/4caa63fe-0308-4e69-96e7-3049d125d643.png) 上，以确定 RNN 预测输出的效果。我们使用交叉熵损失作为我们的损失函数。时间步 ![](img/4caa63fe-0308-4e69-96e7-3049d125d643.png) 处的损失 ![](img/4d9d85c2-78d6-47e5-ae6d-6cd76138ded6.png) 可以如下给出：

![](img/4f24f775-00f1-4b1f-b74f-2b31ea5c2c95.png)

这里，![](img/20982ece-f436-4ca3-be19-332765024b10.png) 是实际输出，而 ![](img/42669ca6-70c1-4d62-b23e-38ba3ecd9cc3.png) 是时间步长为 ![](img/9c19e742-9400-42f5-b91b-c2b833aea576.png) 时的预测输出。

最终损失是所有时间步长上的损失之和。假设我们有 ![](img/81a851a4-6e51-4bec-86de-121b488ed4b1.png) 层；那么，最终损失可以如下给出：

![](img/5ca94542-8bb2-4517-8618-9db73674966f.png)

如下图所示，最终损失是所有时间步长上损失的总和：

![](img/c99e1c4b-b2f2-4885-9c9a-d49b00dd5483.png)

我们计算了损失，现在我们的目标是最小化损失。我们如何最小化损失？我们可以通过找到 RNN 的最优权重来最小化损失。正如我们所学的，RNN 中有三个权重：输入到隐藏的权重 ![](img/506de957-8136-4b7a-b727-16decfa5cb39.png)，隐藏到隐藏的权重 ![](img/6c47a07a-178e-4963-8d71-412cecc002eb.png)，以及隐藏到输出的权重 ![](img/d53286da-5f7e-426d-8412-48e443a6d220.png)。

我们需要找到所有这三个权重的最优值以最小化损失。我们可以使用我们喜爱的梯度下降算法来找到最优权重。我们首先计算损失函数相对于所有权重的梯度，然后根据权重更新规则更新权重，如下所示：

![](img/800d1857-5449-4155-b448-441188297d6c.png)

![](img/a0258154-1ac4-4772-bd1b-7507a7f45b01.png)

![](img/814dc1b3-9460-4875-924e-af775624cc0a.png)

如果您不想理解梯度计算背后的数学，可以跳过接下来的几节。但是，这将有助于您更好地理解循环神经网络中的 BPTT 工作原理。

首先，我们计算损失相对于最终层 ![](img/f451bf26-6e9d-4a0e-8bc3-ee5753ae46e3.png) 的梯度，即 ![](img/f6ece5a6-92ec-4c30-a60e-8b739a0f4728.png)，以便我们可以在接下来的步骤中使用它。

正如我们所学的，时间步骤 ![](img/4caa63fe-0308-4e69-96e7-3049d125d643.png) 处的损失 ![](img/4d9d85c2-78d6-47e5-ae6d-6cd76138ded6.png) 可以表示如下：

![](img/fa612325-7e04-4e72-a3b2-4a1feea9c90f.png)

因为我们知道：

![](img/334d43db-f1c5-4dee-892a-a8ff46c60b0b.png)

我们可以写成：

![](img/001b5467-5c38-4d8c-8cef-e76ec7447d08.png)

因此，损失 ![](img/4d9d85c2-78d6-47e5-ae6d-6cd76138ded6.png) 相对于 ![](img/f451bf26-6e9d-4a0e-8bc3-ee5753ae46e3.png) 的梯度变为：

![](img/68b43156-0c4a-42cd-9fa0-298cbd640074.png)

现在，我们将学习如何逐个计算损失相对于所有权重的梯度。

# 针对隐藏到输出权重 V 的梯度

首先，让我们回顾一下前向传播涉及的步骤：

![](img/cc1a523f-b1b7-4eef-944c-a7cfed5eded2.png)

![](img/14f05246-b094-4bf9-9b08-7b543ecc6b21.png)

![](img/b06b4611-ead7-4a7d-af35-873d64a8da0a.png)

假设 ![](img/c3f72581-aa05-4ad7-b0f0-fbcc278a1597.png)，将其代入方程 *(2)*，我们可以重写上述步骤如下：

![](img/cc1a523f-b1b7-4eef-944c-a7cfed5eded2.png)

![](img/f1d0c5e5-514f-4c19-8a63-f1de866825ae.png)

![](img/855a730d-09e9-4a57-8795-a68202406061.png)

在预测输出 ![](img/f072d0a4-7355-47a7-bea9-9e03ce88bffc.png) 后，我们处于网络的最终层。由于我们正在进行反向传播，即从输出层到输入层，我们的第一个权重将是 ![](img/b5c7341a-7af7-4bb6-9501-174fcfed9b05.png)，即隐藏到输出层的权重。

我们已经看到，最终损失是所有时间步长上的损失之和，类似地，最终梯度是所有时间步长上梯度的总和：

![](img/3403b3f1-f6bb-42ef-b8fa-b236013e5810.png)

因此，我们可以写成：

![](img/9c24ef53-b82f-4f72-81cd-6b7672b1e646.png)

回顾我们的损失函数，![](img/49409524-301d-4a6b-8ec5-f88c6f42d34d.png)；我们不能计算相对于的梯度

![](img/b5c7341a-7af7-4bb6-9501-174fcfed9b05.png)直接来自![](img/842e394d-d8d4-4ad5-9176-0f5415c98d08.png)，因为其中没有![](img/b5c7341a-7af7-4bb6-9501-174fcfed9b05.png)项。因此，我们应用链式法则。回顾前向传播方程；在![](img/db5a20a7-db1c-48b2-85c4-c2017bc7d5ea.png)中有一个![](img/b5c7341a-7af7-4bb6-9501-174fcfed9b05.png)项：

![](img/7dbc66a2-0727-403a-b3a0-bfb1b086ac55.png)，其中![](img/d59f4d43-7dd0-401e-8635-308ebbe3c09b.png)

首先，我们计算损失对![](img/db5a20a7-db1c-48b2-85c4-c2017bc7d5ea.png)的偏导数，然后从![](img/db5a20a7-db1c-48b2-85c4-c2017bc7d5ea.png)中计算对![](img/29d11f59-f2ee-498b-99e9-c318ff3acb9c.png)的偏导数。从![](img/d64191a9-469b-44de-9a17-cc396859ae4f.png)中，我们可以计算对![](img/4f1f59f8-4b8e-43fb-9c5c-baed623ee9d0.png)的导数。

因此，我们的方程如下：

![](img/4af89994-d2cd-45c6-a152-e46b5f05c784.png)

由于我们知道![](img/002f1a76-e814-460a-8c78-4561df5d3a3c.png)，损失函数对![](img/1a5006ba-52b3-49bd-9d8d-476242f5f8bd.png)的梯度可以计算如下：

![](img/4d4684ca-b4ba-4bca-9a04-64bd59d15b02.png)

将方程*(4)*代入方程*(3)*，我们可以写成以下形式：

![](img/36da3d10-1767-4597-b24d-c8833abb4e0c.png)

为了更好地理解，让我们逐个从前述方程中取出每个项并逐个计算：

![](img/1112b7f2-339b-4ce5-99cf-c6df78d40194.png)

根据方程*(1)*，我们可以将![](img/e195ec7d-8c38-4f52-ab57-299b622f2e7c.png)的值代入前述方程*(6)*中，如下所示：

![](img/b2ed4f38-4b08-4015-b1c8-87840c3dfecb.png)

现在，我们将计算项![](img/d09b3f31-d03f-4b17-ba13-fca7055c0054.png)。因为我们知道![](img/002f1a76-e814-460a-8c78-4561df5d3a3c.png)，计算![](img/0bdacbe8-9489-4ef0-bfe8-6ace5c71c78a.png)给出 softmax 函数的导数：

![](img/4e2582cf-bc25-4eb8-b235-d018c1f34ff5.png)

softmax 函数的导数可以表示如下：

![](img/892c67de-449c-4a37-b8f7-75184b88a427.png)

将方程*(8)*代入方程*(7)*，我们可以写成以下形式：

![](img/d7f8d2eb-3cf1-4339-8427-b991003ade3b.png)

因此，最终方程如下：

![](img/32da4bf8-3349-4236-a6a6-43edcdbbe1a8.png)

现在，我们可以将方程*(9)*代入方程*(5)*：

![](img/12399086-bdfa-400a-a02a-5242d04abef7.png)

因为我们知道![](img/2b93d32d-50de-4e91-ab73-7f19e1b80d5f.png)，我们可以写成：

![](img/147a2feb-72a0-4df3-9025-8515df5d35f3.png)

将前述方程代入方程*(10)*，我们得到我们的最终方程，即损失函数对![](img/a1d9a457-be7b-4445-b6f8-9faa63a53991.png)的梯度如下：

![](img/2792d04c-bb47-4e78-a0db-87b4da96e897.png)

# 对隐藏到隐藏层权重 W 的梯度

现在，我们将计算损失相对于隐藏到隐藏层权重 ![](img/a7804dd9-7a32-4507-a4fb-7ea08209578a.png) 的梯度。与 ![](img/683d0de8-d8cd-4798-a13d-a01c5b5f8c3b.png) 类似，最终的梯度是所有时间步长上梯度的总和：

![](img/fa73b527-33cc-4787-b508-e32652748604.png)

因此，我们可以写成：

![](img/834ff810-d70a-49ba-91d7-0ac9640734b3.png)

首先，让我们计算损失的梯度 ![](img/8d47d734-7c46-4b0d-a8de-f88bc29b3918.png) 对 ![](img/43e7ae91-5ab2-40e1-be89-018ed8d7d013.png) 的导数，即 ![](img/80edec19-56f8-4a43-b0df-f70a1ae6651a.png)。

我们不能直接从中计算 ![](img/8d47d734-7c46-4b0d-a8de-f88bc29b3918.png) 对 ![](img/8904cf4f-986d-416e-a712-e41a3596babd.png) 的导数，因为其中没有 ![](img/37e77c54-a6ac-40fc-9b65-71f35645ea44.png) 项。因此，我们使用链式法则计算损失对 ![](img/bd613dea-48a2-4e9b-a593-44b2f3a1258d.png) 的梯度。让我们重新回顾前向传播方程：

![](img/a282aa7f-7ee2-4af0-9088-657715b4ad39.png)

![](img/388b6d16-c63b-436d-af50-c528d4db225b.png)

![](img/fd3bceaa-01ef-494b-ae67-7154329611bc.png)

首先，我们计算损失 ![](img/66a379fd-243b-4699-9944-e06ebc5d56a6.png) 对 ![](img/92490484-5cb1-4184-a19a-9461fa7a5cb5.png) 的偏导数；然后，从 ![](img/783c76ba-a63a-436e-a231-8dc1035cdd78.png) 开始，计算对 ![](img/7f1ac4cd-8eac-4534-b7e7-d42de1db8c94.png) 的偏导数；然后，从 ![](img/cd675325-add1-4e8f-b6b4-8dccdccef57d.png) 开始，我们可以计算对 *W* 的导数，如下所示：

![](img/d0c93e46-3872-4a8e-be11-43fe60a4d504.png)

现在，让我们计算损失的梯度 ![](img/33bc7f4c-569e-4cf8-8e07-264224669dde.png) 对 ![](img/43e7ae91-5ab2-40e1-be89-018ed8d7d013.png) 的导数，即 ![](img/f869ae3e-cf7d-4699-a1cb-2523617d3ce0.png)。因此，我们再次应用链式法则，得到以下结果：

![](img/ed563d77-3ca1-4e89-86d1-b81dd4a79313.png)

如果您看看前述等式，我们如何计算项 ![](img/21ea5fbb-0b85-4e01-82d4-f36846d11bcb.png)？让我们回顾一下 ![](img/0a8a684f-00ec-4748-85fc-fd8e524e9170.png) 的方程：

![](img/ce9781a3-0336-4136-8261-384a1297d27f.png)

正如您在前述等式中所看到的那样，计算 ![](img/b9759f68-c0a4-4f03-8b86-2252753b5919.png) 取决于 ![](img/542ae61a-c8ac-438c-b7a5-bf89b2879073.png) 和 ![](img/e056a8c4-f39b-41d3-b99b-158cfd4aee25.png)，但 ![](img/eb66b04b-0700-4165-bd54-a620f8a087f2.png) 并非常数；它再次是一个函数。因此，我们需要计算其相对于该函数的导数。

然后方程变为：

![](img/825d9ade-e1eb-46cd-9d17-5fcc62897e78.png)

下图显示了计算 ![](img/cfe99387-87de-4d39-a1e3-1c71ee698df0.png)；我们可以注意到 ![](img/b9759f68-c0a4-4f03-8b86-2252753b5919.png) 如何依赖于 ![](img/542ae61a-c8ac-438c-b7a5-bf89b2879073.png)：

![](img/4e1bf1cd-aaa4-45b4-bd1f-de34cde53b66.png)

现在，让我们计算损失函数的梯度，即![](img/b29c042b-b5fd-4bca-943e-7a60c587ae3c.png)关于![](img/43e7ae91-5ab2-40e1-be89-018ed8d7d013.png)的梯度。因此，我们再次应用链式法则，得到以下结果：

![](img/57d5989d-7fd4-4a18-bca9-f72f9992dfeb.png)

在前述方程中，我们无法直接计算![](img/13eb2602-1011-485f-aff3-8aa10fb6240b.png)。回顾方程![](img/58ebb286-48f5-45dd-a933-5ea3149dd2e7.png)：

![](img/0754e778-2993-4050-af9e-2faf72fa6c24.png)

正如您所观察到的，计算![](img/6bdcabae-12f4-491b-b6b7-567c74347953.png)取决于一个函数![](img/5bb0190a-05bc-4728-b2fa-5fe015b3f619.png)，而![](img/6098d4f4-a409-400f-b305-d62ff7843ecd.png)再次是取决于函数![](img/fbcd1328-a49d-4539-823f-6b8f6cdf2896.png)的函数。如下图所示，为了计算关于![](img/a4f53c1b-7611-4c78-b885-56165fe7d195.png)的导数，我们需要遍历直到![](img/59e989bb-1031-471d-9268-8d0db34e2fd1.png)，因为每个函数彼此依赖：

![](img/3d415da8-a11d-42ed-8503-b7aabb36ec9f.png)

这可以用下图来形象地表示：

![](img/ab170b47-574e-412b-8189-1c5d66dc8d2c.png)

这适用于任何时间步长的损失；比如说，![](img/9166e372-27b3-4c05-9ab9-3a5ccf45375b.png)。因此，我们可以说，要计算任何损失![](img/ad3e986a-0ea8-40a6-9494-609395378d07.png)，我们需要遍历到![](img/a2438430-3683-4d72-b56d-99234f9d1cf5.png)，如下图所示：

![](img/e1767315-ee84-4ec5-8a85-0374dba241fe.png)

这是因为在循环神经网络中，时间![](img/d586504e-cac1-4947-b860-904898c59ff5.png)的隐藏状态取决于时间![](img/2883488b-b02f-4f7f-a26a-5a4d4fffa23c.png)的隐藏状态，这意味着当前隐藏状态始终依赖于先前的隐藏状态。

因此，任何损失![](img/6a2f37b3-d174-4890-9382-74e22ebacb22.png)可以如下图所示地计算：

![](img/7fc6c457-3cba-4e88-b68b-46f92a8cf7fa.png)

因此，我们可以写出损失函数![](img/8881eb0a-2a3a-4b4c-8480-0658def14f7b.png)关于![](img/29bb39fe-99a0-4581-84ac-72db22f872b8.png)的梯度如下：

![](img/c1a45c32-3769-4569-8c98-5b07a1c05a75.png)

在前述方程中，前述方程中的总和![](img/110ac4e4-8707-4023-8f35-ab38a73909e0.png)意味着所有隐藏状态![](img/6152d3ec-d37c-43e7-8931-faa1911f6a97.png)的总和。在前述方程中，![](img/6a56cd8e-4322-4688-a2cf-c949c2a3f762.png)可以使用链式法则计算。因此，我们可以说：

![](img/3a596778-bf70-4139-8c90-425ece0b98c6.png)

假设*j=3*和*k=0*；那么，前述方程变为：

![](img/a443c6f9-bea7-4bc1-880f-88156a7847e2.png)

将方程*(12)*代入方程*(11)*将得到以下结果：

![](img/65b1d140-81d3-4055-b111-96f88f4cd367.png)

我们知道最终损失是所有时间步长上损失的总和：

![](img/834ff810-d70a-49ba-91d7-0ac9640734b3.png)

将方程 *(13)* 代入前述方程，我们得到以下结果：

![](img/35c4066f-7230-4ff4-9faf-de6020044c45.png)

在前述方程中，我们有两个求和，其中：

+   ![](img/f3e48fee-b0a7-487e-b1b9-f8ce81aa91ab.png) 暗示了所有时间步长上损失的总和

+   ![](img/753ac5f6-1add-4ee5-b5aa-ece8312eeb5c.png) 是隐藏状态的总和

因此，我们计算损失关于 *W* 的梯度的最终方程为：

![](img/57d629f2-fb47-40df-b50b-a1b58a73e30d.png)

现在，我们将逐一看如何计算上述方程中的每个术语。从方程 *(4)* 和方程 *(9)*，我们可以说：

![](img/3572a974-3b10-4e5b-acb3-57b098b2b847.png)

让我们看下一个术语：

![](img/b2a0ae76-ae52-49ef-a000-c44831b7ebad.png)

我们知道隐藏状态 ![](img/13e1f88c-3dd3-4e1a-9dcf-a4096cc7c42c.png) 的计算为：

![](img/4dc4fda6-8b66-4eaa-8aee-3839088e1cdc.png)

![](img/55d602d9-9548-4859-a99b-dbf12e12ff39.png) 的导数为 ![](img/5ea21d93-8420-4e6a-bf5d-4449fc4d5f6b.png)，因此我们可以写成：

![](img/7f8921c9-8713-4737-88bd-5ea19ea5efce.png)

让我们来看最后一个术语 ![](img/bd407262-7e2f-47b2-a6ef-6cf798693488.png)。我们知道隐藏状态 ![](img/860d20ad-deda-49ed-a11e-b5917198f42d.png) 的计算如下： ![](img/49b5386f-6fd3-498c-956a-6908a8043de2.png)。因此，损失 ![](img/359883d1-8cfb-4c60-b23d-3e487cf7ab5e.png) 关于 ![](img/f63926bf-8e02-4bae-b9e2-a3aa83f268b9.png) 的导数为：

![](img/6f24c981-c8f1-4e1c-9630-63a9ce2f4e2e.png)

将所有计算出的项代入方程 *(15)*，我们得到了关于损失梯度 ![](img/0780e980-493b-4b07-9019-e9935310071a.png) 关于 ![](img/8df7ed9c-b546-4548-86f1-7d99e85b2e19.png) 的最终方程如下：

![](img/00798e74-09a6-4521-a5cd-871cea5e4fd0.png)

# 关于隐藏层权重输入的梯度 U

计算损失函数对 ![](img/7fe2825b-472c-47f5-af2e-7421c1daf9e8.png) 的梯度与 ![](img/470b648e-ccd4-4697-a8a4-7d8fcc9d9551.png) 相同，因为这里我们也是对 ![](img/4447db7f-708c-451a-a72b-9ac034b5e669.png) 进行顺序导数。类似于 ![](img/470b648e-ccd4-4697-a8a4-7d8fcc9d9551.png)，为了计算任何损失 ![](img/24cccd52-50ce-47c3-b94c-9add1dd31f8d.png) 关于 ![](img/7fe2825b-472c-47f5-af2e-7421c1daf9e8.png) 的导数，我们需要沿着一直回到 ![](img/a50a20c6-c54c-4945-b035-6c067baade2f.png)。

计算损失关于 ![](img/7fe2825b-472c-47f5-af2e-7421c1daf9e8.png) 的梯度的最终方程如下。正如你所注意到的那样，它基本上与方程 *(15)* 相同，只是我们有项 ![](img/b7e741dd-bdee-4447-8e4a-2261b07b5691.png) 而不是显示为 ![](img/2e46c8c1-938f-4aec-9ec6-c1dadc1000da.png)：

![](img/3897b39f-55b0-40d4-bcba-3987878caefd.png)

我们已经在前一节中看到如何计算前两项。

让我们看看最终项 ![](img/a4717c74-afdd-4e8d-a388-c577f10db618.png)。我们知道隐藏状态 ![](img/c933d30a-7bf8-4843-b820-d778a186bb1d.png) 计算如下，![](img/0481fc80-3cae-4136-ac58-7f1739f18461.png)。因此，![](img/d6780a9b-59c9-4e73-953f-fa5345662113.png) 对 ![](img/95d442e9-23d4-4676-8cc2-0f70c1c63856.png) 的导数推导如下：

![](img/48384cfd-e122-42ea-8eea-8c1a35f179bf.png)

因此，我们对损失 ![](img/f8f95855-69c2-490c-acf9-68e5f8f67b4d.png) 对 ![](img/7fe2825b-472c-47f5-af2e-7421c1daf9e8.png) 的最终梯度方程可以写成如下形式：

![](img/54561de8-c032-4693-a547-54bf6bd4dfa4.png)

# 梯度消失和梯度爆炸问题

我们刚刚学习了 BPTT 的工作原理，看到了如何计算 RNN 中所有权重的损失梯度。但在这里，我们将遇到一个称为**梯度消失和梯度爆炸**的问题。

在计算损失对 ![](img/817f9d6a-c803-4d12-90b6-1b84f825008d.png) 和 ![](img/8a942335-a0e7-4678-a66f-d90787932787.png) 的导数时，我们看到我们必须遍历直到第一个隐藏状态，因为每个隐藏状态 ![](img/5bb76da6-1ea3-4e61-b400-74a19d5a493c.png) 都依赖于其前一个隐藏状态 ![](img/1a3b3b0a-b675-4a09-9c42-713da17889be.png)。

例如，损失 ![](img/81827407-78c5-447c-b5cc-b1617ffb2f48.png) 对 ![](img/6ec3d4eb-b572-43d2-89ae-4922e8af13d5.png) 的梯度给出如下：

![](img/4b8e1b81-6963-40d7-b578-e8cacca890a8.png)

如果你看一下前述方程中的项 ![](img/9994e50e-4aaf-4b82-9c28-f5b280763ad6.png)，我们无法计算导数

关于 ![](img/ad5d13a0-b7c2-4735-9d11-82480eee7d92.png) 对 ![](img/584bca7b-31c7-409e-8dc8-b144d9a6ce32.png) 的直接推导。正如我们所知，![](img/4f6cbf48-6e6c-4064-9b2e-452da085e021.png) 是一个依赖于 ![](img/6fff19d9-7f93-47d6-a614-3fbb31ae214d.png) 和 ![](img/584bca7b-31c7-409e-8dc8-b144d9a6ce32.png) 的函数。因此，我们也需要计算对 ![](img/6fff19d9-7f93-47d6-a614-3fbb31ae214d.png) 的导数。即使 ![](img/9e192206-167c-46d1-8f9e-c5b227d6acfc.png) 也是一个依赖于 ![](img/697ec559-c85c-4034-a304-0e74b1756b6c.png) 和 ![](img/584bca7b-31c7-409e-8dc8-b144d9a6ce32.png) 的函数。因此，我们还需要计算对 ![](img/eb6488d9-3a99-4b4e-ac4b-06abb1408d52.png) 的导数。

如下图所示，为了计算 ![](img/d1d54dae-34a6-47a4-b0a5-6cafa3ea4f7c.png) 的导数，我们需要一直追溯到初始隐藏状态 ![](img/f3cb32d4-6c46-4f3b-ab25-5639d48617d1.png)，因为每个隐藏状态都依赖于其前一个隐藏状态：

![](img/a4b97dab-8a53-4e2e-b775-ac6d0b583384.png)

因此，为了计算任何损失 ![](img/1c9c6492-be3a-4813-a05d-ec12582130d3.png)，我们需要一直回溯到初始隐藏状态。

状态 ![](img/931e2725-bc7b-49f0-a943-42cb9c89b617.png)，因为每个隐藏状态都依赖于其前一个隐藏状态。假设我们有一个具有 50 层的深度递归网络。为了计算损失 ![](img/a23cf5fb-48d2-471b-aca5-766399059c5d.png)，我们需要一直回溯到 ![](img/931e2725-bc7b-49f0-a943-42cb9c89b617.png)，如下图所示：

![](img/4dcd0cd0-a0f7-4f67-b283-83cb90d68d5e.png)

所以，问题究竟出在哪里？在向初始隐藏状态反向传播时，我们丢失了信息，RNN 将不能完美地反向传播。

记得 ![](img/d913a9e9-5fee-4e6e-8139-cb8c1e5ce0c3.png) 吗？每次向后移动时，我们计算 ![](img/33120d82-b5ac-462f-8bae-9e599cb7aba0.png) 的导数。tanh 的导数被限制在 1\. 我们知道，当两个介于 0 和 1 之间的值相乘时，结果将会更小。我们通常将网络的权重初始化为一个小数。因此，当我们在反向传播时乘以导数和权重时，实质上是在乘以较小的数。

当我们在向后移动时，每一步乘以较小的数，我们的梯度变得无限小，导致计算机无法处理的数值；这被称为**梯度消失问题**。

回顾我们在 *关于隐藏层到隐藏层权重 W 的梯度* 部分看到的关于损失的梯度方程式：

![](img/cf4137b0-4a11-4b38-ab29-33b702dced2a.png)

正如你所看到的，我们在每个时间步长上乘以权重和 tanh 函数的导数。这两者的重复乘法会导致一个很小的数，从而引起梯度消失问题。

梯度消失问题不仅出现在 RNN 中，还出现在其他使用 sigmoid 或 tanh 作为激活函数的深层网络中。因此，为了克服这个问题，我们可以使用 ReLU 作为激活函数，而不是 tanh。

然而，我们有一种称为**长短期记忆**（**LSTM**）网络的 RNN 变体，它可以有效地解决梯度消失问题。我们将在第五章 *RNN 的改进* 中看看它是如何工作的。

同样地，当我们将网络的权重初始化为非常大的数时，在每一步中梯度会变得非常大。在反向传播时，我们在每个时间步长上乘以一个大数，导致梯度爆炸。这被称为**梯度爆炸问题**。

# 梯度裁剪

我们可以使用梯度裁剪来避免梯度爆炸问题。在这种方法中，我们根据向量范数（比如 *L2* 范数）来归一化梯度，并将梯度值裁剪到某个范围内。例如，如果我们将阈值设为 0.7，那么我们保持梯度在 -0.7 到 +0.7 的范围内。如果梯度值超过 -0.7，我们将其更改为 -0.7；同样地，如果超过 0.7，我们将其更改为 +0.7。

假设 ![](img/46772caf-a135-4853-b011-cbfe4346843c.png) 是损失函数 L 关于 W 的梯度：

![](img/70ac4730-be3f-48b6-829c-d4cc6ac84a0e.png)

首先，我们使用 L2 范数对梯度进行归一化，即 ![](img/7201c6b6-7ce4-4f30-af35-ee6ea0be0d0f.png)。如果归一化的梯度超过了定义的阈值，我们更新梯度如下：

![](img/0cde0a7c-4f3b-45ec-ad0a-ab169da0b938.png)

# 使用 RNN 生成歌词

我们已经学习了关于 RNN 的足够知识；现在，让我们看看如何使用 RNN 生成歌词。为此，我们简单地构建一个字符级 RNN，也就是说，在每个时间步，我们预测一个新字符。

让我们考虑一个小句子，*What a beautiful d*。

在第一个时间步，RNN 预测一个新字符 *a*。句子将更新为 *What a beautiful* *d**a**.*。

在下一个时间步，它预测一个新字符 *y*，句子变成了 *What a beautiful da**y**.*。

这样，我们每个时间步预测一个新字符并生成一首歌曲。除了每次预测一个新字符外，我们还可以每次预测一个新单词，这称为**词级 RNN**。为简单起见，让我们从字符级 RNN 开始。

但是，RNN 如何在每个时间步预测一个新字符呢？假设在时间步 t=0 时，我们输入一个字符 *x*。现在 RNN 根据给定的输入字符 *x* 预测下一个字符。为了预测下一个字符，它会预测我们词汇表中所有字符成为下一个字符的概率。一旦得到这个概率分布，我们根据这个概率随机选择下一个字符。有点糊涂吗？让我们通过一个例子更好地理解这个过程。

例如，如下图所示，假设我们的词汇表包含四个字符 *L, O, V,* 和 *E*；当我们将字符 *L* 作为输入时，RNN 计算词汇表中所有单词成为下一个字符的概率：

![](img/79adc628-079c-474e-b687-af043cadc304.png)

因此，我们得到概率 **[0.0, 0.9, 0.0, 0.1]**，对应词汇表中的字符 *[L,O,V,E]*。通过从这个概率分布中抽样来预测下一个字符，给输出增加了一些随机性。

在下一个时间步上，我们将上一时间步预测的字符和先前的隐藏状态作为输入，预测下一个字符，如下图所示：

![](img/d44f75dd-af2d-4754-8992-cc3bb20e1793.png)

因此，在每个时间步上，我们将上一时间步预测的字符和先前的隐藏状态作为输入，并预测下一个字符，如下所示：

![](img/2741fb9b-44f2-4e0f-a7b6-07359373bd1f.png)

正如您在前面的图中所见，在时间步 *t=2*，*V* 作为输入传递，并预测下一个字符为 *E*。但这并不意味着每次将字符 *V* 作为输入发送时都应始终返回 *E* 作为输出。由于我们将输入与先前的隐藏状态一起传递，RNN 记住了到目前为止看到的所有字符。

因此，先前的隐藏状态捕捉了前面输入字符的精髓，即 *L* 和 *O*。现在，使用此前的隐藏状态和输入 *V*，RNN 预测下一个字符为 *E*。

# 在 TensorFlow 中实现

现在，我们将看看如何在 TensorFlow 中构建 RNN 模型来生成歌词。该数据集以及本节中使用的完整代码和逐步说明可以在 GitHub 上的 [`bit.ly/2QJttyp`](http://bit.ly/2QJttyp) 获取。下载后，解压缩档案，并将 `songdata.csv` 放在 `data` 文件夹中。

导入所需的库：

```py
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import warnings
warnings.filterwarnings('ignore')
```

# 数据准备

读取下载的输入数据集：

```py
df = pd.read_csv('data/songdata.csv')
```

让我们看看我们的数据集中有什么：

```py
df.head()
```

前述代码生成如下输出：

![](img/b141430d-703a-4651-b047-85272150672d.png)

我们的数据集包含约 57,650 首歌曲：

```py
df.shape[0]

57650
```

我们有约 `643` 位艺术家的歌词：

```py
len(df['artist'].unique())

643
```

每位艺术家的歌曲数量如下所示：

```py
df['artist'].value_counts()[:10]

Donna Summer        191
Gordon Lightfoot    189
George Strait       188
Bob Dylan           188
Loretta Lynn        187
Cher                187
Alabama             187
Reba Mcentire       187
Chaka Khan          186
Dean Martin         186
Name: artist, dtype: int64
```

平均每位艺术家有约 `89` 首歌曲：

```py
df['artist'].value_counts().values.mean()

89
```

我们在 `text` 列中有歌词，因此我们将该列的所有行组合起来，并将其保存为名为 `data` 的变量中的 `text`，如下所示：

```py
data = ', '.join(df['text'])
```

让我们看看一首歌的几行：

```py
data[:369]

"Look at her face, it's a wonderful face  \nAnd it means something special to me  \nLook at the way that she smiles when she sees me  \nHow lucky can one fellow be?  \n  \nShe's just my kind of girl, she makes me feel fine  \nWho could ever believe that she could be mine?  \nShe's just my kind of girl, without her I'm blue  \nAnd if she ever leaves me what could I do, what co"
```

由于我们正在构建字符级 RNN，我们将数据集中所有唯一字符存储在名为 `chars` 的变量中；这基本上就是我们的词汇表：

```py
chars = sorted(list(set(data)))
```

将词汇表大小存储在名为 `vocab_size` 的变量中：

```py
vocab_size = len(chars)
```

由于神经网络只接受数字输入，因此我们需要将词汇表中的所有字符转换为数字。

我们将词汇表中的所有字符映射到它们的对应索引，形成一个唯一的数字。我们定义了一个 `char_to_ix` 字典，其中包含所有字符到它们索引的映射。为了通过字符获取索引，我们还定义了 `ix_to_char` 字典，其中包含所有索引到它们相应字符的映射：

```py
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
```

如您在下面的代码片段中所见，字符 `'s'` 在 `char_to_ix` 字典中映射到索引 `68`：

```py
print char_to_ix['s']

68
```

类似地，如果我们将 `68` 作为输入给 `ix_to_char`，那么我们得到相应的字符，即 `'s'`：

```py
print ix_to_char[68]

's'
```

一旦我们获得字符到整数的映射，我们使用独热编码将输入和输出表示为向量形式。**独热编码向量** 基本上是一个全为 0 的向量，除了对应字符索引位置为 1。

例如，假设 `vocabSize` 是 `7`，而字符 *z* 在词汇表中的第四个位置。那么，字符 *z* 的独热编码表示如下所示：

```py
vocabSize = 7
char_index = 4

print np.eye(vocabSize)[char_index]

array([0., 0., 0., 0., 1., 0., 0.])
```

如您所见，我们在对应字符的索引位置有一个 1，其余值为 0。这就是我们将每个字符转换为独热编码向量的方式。

在以下代码中，我们定义了一个名为 `one_hot_encoder` 的函数，该函数将根据字符的索引返回一个独热编码向量：

```py
def one_hot_encoder(index):
    return np.eye(vocab_size)[index]
```

# 定义网络参数

接下来，我们定义所有网络参数：

1.  定义隐藏层中的单元数：

```py
hidden_size = 100
```

1.  定义输入和输出序列的长度：

```py
seq_length = 25
```

1.  定义梯度下降的学习率：

```py
learning_rate = 1e-1
```

1.  设置种子值：

```py
seed_value = 42
tf.set_random_seed(seed_value)
random.seed(seed_value)
```

# 定义占位符

现在，我们将定义 TensorFlow 的占位符：

1.  输入和输出的 `placeholders` 定义如下：

```py
inputs = tf.placeholder(shape=[None, vocab_size],dtype=tf.float32, name="inputs")
targets = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32, name="targets")
```

1.  定义初始隐藏状态的 `placeholder`：

```py
init_state = tf.placeholder(shape=[1, hidden_size], dtype=tf.float32, name="state")
```

1.  定义用于初始化 RNN 权重的 `initializer`：

```py
initializer = tf.random_normal_initializer(stddev=0.1)
```

# 定义前向传播

让我们定义涉及 RNN 的前向传播，数学表达如下：

![](img/523f1fa0-bea0-4810-9134-da3e0f971538.png)

![](img/5acff446-3893-46e3-b0fb-fe80338957c8.png)

![](img/c4dd726d-98a7-4cfa-bf7d-1b82f8d98c21.png) 和 ![](img/f25b6fa6-b0c4-42e0-b737-89dfec3ec7a2.png) 是隐藏层和输出层的偏置项，简单起见，在前面的方程中我们没有添加它们。前向传播可以实现如下：

```py
with tf.variable_scope("RNN") as scope:
    h_t = init_state
    y_hat = []

    for t, x_t in enumerate(tf.split(inputs, seq_length, axis=0)):
        if t > 0:
            scope.reuse_variables() 

        #input to hidden layer weights
        U = tf.get_variable("U", [vocab_size, hidden_size], initializer=initializer)

        #hidden to hidden layer weights
        W = tf.get_variable("W", [hidden_size, hidden_size], initializer=initializer)

        #output to hidden layer weights
        V = tf.get_variable("V", [hidden_size, vocab_size], initializer=initializer)

        #bias for hidden layer
        bh = tf.get_variable("bh", [hidden_size], initializer=initializer)

        #bias for output layer
        by = tf.get_variable("by", [vocab_size], initializer=initializer)

        h_t = tf.tanh(tf.matmul(x_t, U) + tf.matmul(h_t, W) + bh)

        y_hat_t = tf.matmul(h_t, V) + by

        y_hat.append(y_hat_t) 
```

对输出应用 `softmax` 并获取概率：

```py
output_softmax = tf.nn.softmax(y_hat[-1])
outputs = tf.concat(y_hat, axis=0)
```

计算交叉熵损失：

```py
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=outputs))
```

将 RNN 的最终隐藏状态存储在 `hprev` 中。我们使用这个最终隐藏状态来进行预测：

```py
hprev = h_t
```

# 定义 BPTT

现在，我们将执行 BPTT，使用 Adam 作为优化器。我们还将进行梯度裁剪以避免梯度爆炸问题：

1.  初始化 Adam 优化器：

```py
minimizer = tf.train.AdamOptimizer()
```

1.  使用 Adam 优化器计算损失的梯度：

```py
gradients = minimizer.compute_gradients(loss)
```

1.  设置梯度裁剪的阈值：

```py
threshold = tf.constant(5.0, name="grad_clipping")
```

1.  裁剪超过阈值的梯度并将其带入范围内：

```py
clipped_gradients = []
for grad, var in gradients:
    clipped_grad = tf.clip_by_value(grad, -threshold, threshold)
    clipped_gradients.append((clipped_grad, var))
```

1.  使用裁剪后的梯度更新梯度：

```py
updated_gradients = minimizer.apply_gradients(clipped_gradients)
```

# 开始生成歌曲

开始 TensorFlow 会话并初始化所有变量：

```py
sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)
```

现在，我们将看看如何使用 RNN 生成歌词。RNN 的输入和输出应该是什么？它是如何学习的？训练数据是什么？让我们逐步理解这一解释，并附上代码。

我们知道在 RNN 中，在时间步 ![](img/db822ea3-1b38-46ec-bb99-2663c951b137.png) 预测的输出将作为下一个时间步的输入；也就是说，每个时间步，我们需要将前一个时间步预测的字符作为输入。因此，我们以同样的方式准备我们的数据集。

例如，请看下表。假设每一行代表一个不同的时间步；在时间步 ![](img/0249a9fb-0d9b-4b2f-93ff-4b22011a9441.png)，RNN 预测一个新字符 *g* 作为输出。这将作为下一个时间步 ![](img/8556fd17-b301-4584-8a9a-1d06c3082bc6.png) 的输入发送。

然而，如果你注意到时间步 ![](img/8556fd17-b301-4584-8a9a-1d06c3082bc6.png) 中的输入，我们从输入 `o` 中删除了第一个字符，并在序列末尾添加了新预测的字符 *g*。为什么要删除输入中的第一个字符？因为我们需要保持序列长度。

假设我们的序列长度是八；将新预测的字符添加到序列中会将序列长度增加到九。为了避免这种情况，我们从输入中移除第一个字符，同时将前一个时间步的新预测字符添加进来。

类似地，在输出数据中，我们也会在每个时间步删除第一个字符，因为一旦预测了新字符，序列长度就会增加。为了避免这种情况，在每个时间步从输出中删除第一个字符，如下表所示：

![](img/4a4c6500-b69d-4016-a58b-91988ee76b16.png)

现在，我们将看看如何准备我们的输入和输出序列，类似于前面的表格。

定义一个名为 `pointer` 的变量，它指向数据集中的字符。我们将 `pointer` 设置为 `0`，这意味着它指向第一个字符：

```py
pointer = 0
```

定义输入数据：

```py
input_sentence = data[pointer: pointer + seq_length]
```

这是什么意思？使用指针和序列长度切片数据。假设 `seq_length` 是 `25`，指针是 `0`。它将返回前 25 个字符作为输入。因此，`data[pointer:pointer + seq_length]` 返回以下输出：

```py
"Look at her face, it's a "
```

定义输出，如下所示：

```py
output_sentence = data[pointer + 1: pointer + seq_length + 1]
```

我们将输出数据切片，从输入数据移动一个字符。因此，`data[pointer + 1:pointer + seq_length + 1]` 返回以下内容：

```py
"ook at her face, it's a w"
```

正如你所看到的，我们在前一句中添加了下一个字符并删除了第一个字符。因此，每次迭代时，我们增加指针并遍历整个数据集。这就是我们为训练 RNN 获取输入和输出句子的方法。

就像我们学到的那样，RNN 仅接受数字作为输入。一旦我们切片了输入和输出序列，我们使用之前定义的 `char_to_ix` 字典获取相应字符的索引：

```py
input_indices = [char_to_ix[ch] for ch in input_sentence]
target_indices = [char_to_ix[ch] for ch in output_sentence]
```

使用我们之前定义的 `one_hot_encoder` 函数将索引转换为 one-hot 编码向量：

```py
input_vector = one_hot_encoder(input_indices)
target_vector = one_hot_encoder(target_indices)
```

这 `input_vector` 和 `target_vector` 成为训练 RNN 的输入和输出。现在，让我们开始训练。

`hprev_val`变量存储了我们训练的 RNN 模型的最后隐藏状态，我们用它来进行预测，并将损失存储在名为`loss_val`的变量中：

```py
hprev_val, loss_val, _ = sess.run([hprev, loss, updated_gradients], feed_dict={inputs: input_vector,targets: target_vector,init_state: hprev_val})
```

我们训练模型进行*n*次迭代。训练后，我们开始进行预测。现在，我们将看看如何进行预测并使用我们训练的 RNN 生成歌词。设置`sample_length`，即我们想要生成的句子（歌曲）的长度：

```py
sample_length = 500
```

随机选择输入序列的起始索引：

```py
random_index = random.randint(0, len(data) - seq_length)
```

选择具有随机选择索引的输入句子：

```py
sample_input_sent = data[random_index:random_index + seq_length]
```

正如我们所知，我们需要将输入作为数字进行馈送；将所选输入句子转换为索引：

```py
sample_input_indices = [char_to_ix[ch] for ch in sample_input_sent]
```

记住，我们将 RNN 的最后隐藏状态存储在`hprev_val`中。我们使用它来进行预测。我们通过从`hprev_val`复制值来创建一个名为`sample_prev_state_val`的新变量。

`sample_prev_state_val`用作进行预测的初始隐藏状态：

```py
sample_prev_state_val = np.copy(hprev_val)
```

初始化用于存储预测输出索引的列表：

```py
predicted_indices = []
```

现在，对于`t`在`sample_length`的范围内，我们执行以下操作并为定义的`sample_length`生成歌曲：

将`sampled_input_indices`转换为 one-hot 编码向量：

```py
sample_input_vector = one_hot_encoder(sample_input_indices)
```

将`sample_input_vector`和初始隐藏状态`sample_prev_state_val`馈送给 RNN，并获得预测。我们将输出概率分布存储在`probs_dist`中：

```py
probs_dist, sample_prev_state_val = sess.run([output_softmax, hprev],
 feed_dict={inputs: sample_input_vector,init_state: sample_prev_state_val})
```

使用由 RNN 生成的概率分布随机选择下一个字符的索引：

```py
ix = np.random.choice(range(vocab_size), p=probs_dist.ravel())
```

将新预测的索引`ix`添加到`sample_input_indices`中，并从`sample_input_indices`中删除第一个索引以保持序列长度。这将形成下一个时间步的输入：

```py
sample_input_indices = sample_input_indices[1:] + [ix]
```

存储所有预测的`chars`索引在`predicted_indices`列表中：

```py
predicted_indices.append(ix)
```

将所有的`predicted_indices`转换为它们对应的字符：

```py
predicted_chars = [ix_to_char[ix] for ix in predicted_indices]
```

将所有的`predicted_chars`组合起来，并保存为`text`：

```py
 text = ''.join(predicted_chars)
```

在每 50,000^(th)次迭代时打印预测文本：

```py
print ('\n')
print (' After %d iterations' %(iteration))
print('\n %s \n' % (text,)) 
print('-'*115)
```

增加`pointer`和`iteration`：

```py
pointer += seq_length
iteration += 1
```

在初始迭代中，您可以看到 RNN 生成了随机字符。但是在第 50,000^(th)次迭代中，它开始生成有意义的文本：

```py
 After 0 iterations

 Y?a6C.-eMSfk0pHD v!74YNeI 3YeP,h- h6AADuANJJv:HA(QXNeKzwCjBnAShbavSrGw7:ZcSv[!?dUno Qt?OmE-PdY wrqhSu?Yvxdek?5Rn'Pj!n5:32a?cjue  ZIj
Xr6qn.scqpa7)[MSUjG-Sw8n3ZexdUrLXDQ:MOXBMX EiuKjGudcznGMkF:Y6)ynj0Hiajj?d?n2Iapmfc?WYd BWVyB-GAxe.Hq0PaEce5H!u5t: AkO?F(oz0Ma!BUMtGtSsAP]Oh,1nHf5tZCwU(F?X5CDzhOgSNH(4Cl-Ldk? HO7 WD9boZyPIDghWUfY B:r5z9Muzdw2'WWtf4srCgyX?hS!,BL GZHqgTY:K3!wn:aZGoxr?zmayANhMKJsZhGjpbgiwSw5Z:oatGAL4Xenk]jE3zJ?ymB6v?j7(mL[3DFsO['Hw-d7htzMn?nm20o'?6gfPZhBa
NlOjnBd2n0 T"d'e1k?OY6Wwnx6d!F 

----------------------------------------------------------------------------------------------

 After 50000 iterations

 Hem-:]  
[Ex" what  
Akn'lise  
[Grout his bring bear.  
Gnow ourd?  
Thelf  
As cloume  
That hands, Havi Musking me Mrse your leallas, Froking the cluse (have: mes.  
I slok and if a serfres me the sky withrioni flle rome.....Ba tut get make ome  
But it lives I dive.  
[Lett it's to the srom of and a live me it's streefies  
And is.  
As it and is me dand a serray]  
zrtye:"  
Chay at your hanyer  
[Every rigbthing with farclets  

[Brround.  
Mad is trie  
[Chare's a day-Mom shacke?

, I  

-------------------------------------------------------------------------------------------------
```

# 不同类型的 RNN 架构

现在我们已经了解了 RNN 的工作原理，我们将看看基于输入和输出数量的不同类型的 RNN 架构。

# 一对一架构

在**一对一架构**中，单个输入映射到单个输出，时间步*t*的输出作为下一个时间步的输入。我们已经在最后一节中看到了这种架构，用于使用 RNN 生成歌曲。

例如，在文本生成任务中，我们将当前时间步生成的输出作为下一个时间步的输入，以生成下一个单词。这种架构在股票市场预测中也被广泛使用。

下图展示了一对一的 RNN 架构。如您所见，时间步*t*预测的输出被发送为下一个时间步的输入：

![# 一对多架构在**一对多架构**中，单个输入映射到多个隐藏状态和多个输出值，这意味着 RNN 接受单个输入并映射到输出序列。尽管我们只有一个输入值，但我们在时间步之间共享隐藏状态以预测输出。与前述一对一架构不同，这里我们只在时间步之间共享上一个隐藏状态，而不是上一个输出。这种架构的一个应用是图像标题生成。我们传递单个图像作为输入，输出是构成图像标题的单词序列。如下图所示，将单个图像作为输入传递给 RNN，在第一个时间步，预测单词*Horse*；在下一个时间步，使用上一个隐藏状态预测下一个单词*standing*。类似地，它持续一系列步骤并预测下一个单词，直到生成标题：![](img/e898e507-4926-4708-84b4-3fd31fa19b80.png)

# 多对一架构

一个**多对一架构**，顾名思义，接受一个输入序列并将其映射到单个输出值。多对一架构的一个流行示例是**情感分类**。一句话是一个单词序列，因此在每个时间步，我们将每个单词作为输入传递，并在最终时间步预测输出。

假设我们有一个句子：*Paris is a beautiful city.* 如下图所示，在每个时间步，一个单词作为输入传递，并且在最终时间步预测句子的情感：

![](img/922ee723-6f74-4265-b9fb-8837d4c7ee16.png)

# **多对多架构**

在**多对多架构**中，我们将任意长度的输入序列映射到任意长度的输出序列。这种架构已被用于各种应用中。多对多架构的一些流行应用包括语言翻译、对话机器人和音频生成。

假设我们正在将一句英语句子转换成法语。考虑我们的输入句子：*What are you doing?* 如下图所示，它将映射为*Que faites vous*：

![](img/52a6c838-3b48-48e6-a775-46af909f8fb1.png)

# 总结

我们首先介绍了什么是 RNN，以及 RNN 与前馈网络的区别。我们了解到 RNN 是一种特殊类型的神经网络，广泛应用于序列数据；它根据当前输入和先前的隐藏状态预测输出，隐藏状态充当记忆，存储到目前为止网络所见到的信息序列。

我们学习了 RNN 中的前向传播工作原理，然后详细推导了用于训练 RNN 的 BPTT 算法。接着，我们通过在 TensorFlow 中实现 RNN 来生成歌词。在本章末尾，我们了解了 RNN 的不同架构，如一对一、一对多、多对一和多对多，它们被用于各种应用中。

在下一章中，我们将学习关于 LSTM 单元的内容，它解决了 RNN 中的梯度消失问题。我们还将学习不同变体的 RNN。

# 问题

尝试回答以下问题：

1.  RNN 与前馈神经网络有什么区别？

1.  RNN 中的隐藏状态是如何计算的？

1.  递归网络有什么用途？

1.  梯度消失问题是如何发生的？

1.  什么是爆炸梯度问题？

1.  梯度裁剪如何缓解爆炸梯度问题？

1.  不同类型的 RNN 架构有哪些？

# 进一步阅读

参考以下链接了解更多关于 RNN 的内容：

+   *循环神经网络（RNN）和长短期记忆（LSTM）网络基础*，作者 Alex Sherstinsky，[`arxiv.org/pdf/1808.03314.pdf`](https://arxiv.org/pdf/1808.03314.pdf)

+   利用 TensorFlow 进行手写生成的 RNN，基于 Alex Graves 的*用循环神经网络生成序列*，[`github.com/snowkylin/rnn-handwriting-generation`](https://github.com/snowkylin/rnn-handwriting-generation)
