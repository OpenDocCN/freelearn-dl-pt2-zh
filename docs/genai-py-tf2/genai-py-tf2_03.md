# 第三章：深度神经网络的构建模块

在本书中，我们将实现的广泛范围的生成式人工智能模型都是建立在过去十年来在*深度学习*和神经网络方面的进步基础上的。虽然在实践中我们可以在不参考历史发展的情况下实现这些项目，但追溯它们的基本组成部分将使您对这些模型*如何*和*为什么*工作有更深入的理解。在本章中，我们将深入探讨这一背景，向您展示生成式人工智能模型是如何从基础构建的，如何将较小的单元组装成复杂的架构，这些模型中的损失函数是如何优化的，以及一些当前的理论解释为什么这些模型如此有效。掌握了这些背景知识，您应该能够更深入地理解从本书的第四章开始的更高级模型和主题背后的推理，《教网络生成数字》。一般来说，我们可以将神经网络模型的构建模块分为一些关于模型如何构建和训练的选择，我们将在本章中进行介绍：

使用哪种神经网络架构：

+   感知器

+   **多层感知器**（**MLP**）/ 前馈

+   **卷积神经网络**（**CNNs**）

+   **循环神经网络**（**RNNs**）

+   **长短期记忆网络**（**LSTMs**）

+   **门控循环单元**（**GRUs**）

在网络中使用哪些激活函数：

+   线性

+   Sigmoid

+   Tanh

+   ReLU

+   PReLU

使用什么优化算法来调整网络参数：

+   **随机梯度下降**（**SGD**）

+   RMSProp

+   AdaGrad

+   ADAM

+   AdaDelta

+   无 Hessian 优化

如何初始化网络的参数：

+   随机

+   Xavier 初始化

+   He 初始化

正如你所理解的，这些决策的产物可能导致大量潜在的神经网络变种，开发这些模型的一个挑战之一是确定每个选择中的正确搜索空间。在描述神经网络历史的过程中，我们将更详细地讨论这些模型参数的影响。我们对这个领域的概述始于这一学科的起源：谦逊的感知器模型。

# 感知器——一个功能中的大脑

最简单的神经网络架构——感知器——受生物研究的启发，旨在理解心理加工的基础，试图用数学公式表示大脑的功能。在本节中，我们将涵盖一些早期研究，以及它是如何激发了现在的深度学习和生成式人工智能领域的。

## 从组织到 TLUs

AI 算法的近期受欢迎可能会给人一种错误的印象，认为这个领域是新的。许多近期的模型基于几十年前的发现，这些发现因云端的大规模计算资源以及用于并行矩阵计算的定制硬件（如**图形处理单元**（**GPUs**）、**张量处理单元**（**TPUs**）和**可编程门阵列**（**FPGAs**））而得到了重振。如果我们认为神经网络的研究包括其生物启发和计算理论，那么这个领域已经有上百年的历史了。事实上，19 世纪科学家 Santiago Ramón y Cajal 详细解剖插图中描述的其中一个最早的神经网络，这些插图基于对相互连接的神经细胞层的实验观察，启发了神经元学说—即大脑是由单独的、物理上不同且专门的细胞组成，而不是一个连续的网络。¹ Cajal 观察到的视网膜的不同层也启发了特定的神经网络架构，比如我们将在本章后面讨论的 CNN。

![](img/B16176_03_01.png)

图 3.1：由 Santiago Ramón y Cajal 绘制的神经元相互连接的网络³

这种简单神经细胞相互连接的观察使得计算研究人员推测精神活动可能如何由简单的逻辑运算表示，进而产生复杂的精神现象。最初的“自动机理论”通常被追溯到麻省理工学院的 Warren McCulloch 和 Walter Pitts 于 1943 年发表的一篇文章。³ 他们描述了一个简单的模型，即**阈值逻辑单元**（**TLU**），其中二进制输入根据阈值转换为二进制输出：

![](img/B16176_03_001.png)

其中，*I* 代表输入值，*W* 代表权重范围为 (0, 1) 或 (-1, 1)，而 f 是一个阈值函数，根据输入是否超过阈值 *T* 将这些输入转换成二进制输出：⁴

![](img/B16176_03_002.png)

在视觉上和概念上，McCulloch 和 Pitts 的模型与启发它的生物神经元（*图 3.2*）之间存在一定的相似性。他们的模型将输入整合成输出信号，就像神经元的自然树突（神经元的短输入“臂”，从其他细胞接收信号）将输入通过轴突（细胞的长“尾巴”，将从树突接收到的信号传递给其他神经元）合成一个单一的输出。我们可以想象，就像神经细胞被组成网络以产生复杂的生物学电路一样，这些简单的单元可能被连接起来以模拟复杂的决策过程。

![](img/B16176_03_02_a+b.png)

图 3.2：TLU 模型和生物神经元^(5 6)

实际上，使用这个简单的模型，我们已经可以开始表示几个逻辑操作。 如果我们考虑一个带有一个输入的简单神经元的情况，我们可以看到 TLU 可以解决恒等或否定函数（*表 3.1*和*3.2*）。

对于一个简单地返回输入作为输出的恒等操作，权重矩阵在对角线上会有 1（或者对于单个数字输入，权重矩阵会简单地是标量 1，如*表 1*所示）：

| 恒等 |
| --- |
| 输入 | 输出 |
| 1 | 1 |
| 0 | 0 |

Table 3.1：恒等操作的 TLU 逻辑

同样地，对于否定操作，权重矩阵可以是负对角线矩阵，在阈值为 0 时翻转输出的符号：

| 否定 |
| --- |
| 输入 | 输出 |
| 1 | 0 |
| 0 | 1 |

Table 3.2：否定操作的 TLU 逻辑

给定两个输入，TLU 也可以表示诸如 AND 和 OR 等操作。 在这里，可以设置一个阈值，使得组合输入值必须超过`2`（对应 AND 操作的输出为`1`）或者`1`（对应 OR 操作的输出为`1`，如果两个输入中任意一个为`1`）。

| AND |
| --- |
| 输入 1 | 输入 2 | 输出 |
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 1 | 1 |

Table 3.3：AND 操作的 TLU 逻辑

| OR |
| --- |
| 输入 1 | 输入 2 | 输出 |
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 1 |

Table 3.4：OR 操作的 TLU 逻辑

然而，TLU 无法捕获诸如“异或”（XOR）的模式，它只有在`OR`条件为真时才会输出`1`。

| 异或 |
| --- |
| 输入 1 | 输入 2 | 输出 |
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |

Table 3.5：XOR 操作的 TLU 逻辑

要看到这为什么是真的，考虑一个有两个输入和正权重值为`1`的 TLU。 如果阈值值`T`为`1`，那么输入(`0`, `0`)，(`1`, `0`)和(`0`, `1`)将产生正确的值。 然而，(`1`, `1`)会发生什么？ 因为阈值函数对于任何求和大于`1`的输入都返回`1`，它无法表示 XOR（*表 3.5*），因为 XOR 要求一旦超过不同的、更高的值，就要计算不同的输出。 改变一个或两个权重为负值也没有帮助； 问题在于决策阈值只能单向操作，不能对更大的输入进行反转。

同样地，TLU 不能表示“异或”的否定，即`XNOR`（*表 3.6*）。

| XNOR |
| --- |
| 输入 1 | 输入 2 | 输出 |
| 0 | 0 | 1 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 1 | 1 |

Table 3.6：XNOR 操作的 TLU 逻辑

与`XOR`操作类似（*表 3.5*），通过考虑一个含有两个 1 的权重矩阵，可以说明无法通过 TLU 函数来表示`XNOR`操作（*表 3.6*）；对于两个输入（1, 0）或（0, 1），如果我们设置输出 1 的阈值为 2，则获得了正确的值。与`XOR`操作类似，当输入为（0, 0）时会遇到问题，因为我们无法设置第二个阈值来使和为 0 时输出 1。

## 从 TLUs 到调谐感知器

除了这些有关表示`XOR`和`XNOR`操作的限制外，还有一些附加的简化会限制 TLU 模型的表达能力；权重是固定的，输出只能是二进制（0 或 1）。显然，对于像神经元这样的系统来说，“学习”需要对环境做出响应，并根据先前的经验反馈确定不同输入的相关性。这个观点在加拿大心理学家唐纳德·赫布（Donald Hebb）1949 年的著作《行为的组织》中有所体现，他提出，附近的神经细胞的活动随着时间会趋同，有时被简化为赫布定律：“放电在一起联结在一起”^(7 8)。基于赫布的权重随时间变化的提议，康奈尔航空实验室的研究员弗兰克·罗森布拉特（Frank Rosenblatt）在 1950 年代提出了感知器（perceptron）模型。⁹ 他用自适应权重替代了 TLU 模型中的固定权重，并增加了偏置项，得到了一个新的函数：

![](img/B16176_03_003.png)

我们注意到，输入*I*已被标记为*X*以突显它们可以是任何值，而不仅仅是二进制`0`或`1`。将赫布的观察与 TLU 模型相结合，感知器的权重将根据简单的学习规则进行更新：

1.  从一组 J 个样本*x*(1) …. x(*j*)出发。这些样本都有标签 y，可以是 0 或 1，提供有标记的数据（*y*, *x*)(1) …. (*y*, *x*)(*j*)。这些样本可以是单个值，此时感知器有单个输入，也可以是长度为*N*且具有* i*的多值输入的向量。

1.  初始化所有权重*w*为小的随机值或 0。

1.  使用感知器函数计算所有示例*x*的估计值*yhat*。

1.  使用学习速率*r*更新权重，以更接近于每一步*t*中训练的期望输出值：

    ![](img/B16176_03_004.png)，对于所有的*J*个样本和*N*个特征。概念上，需要注意如果*y*为 0 且目标值为 1，我们希望通过一定的增量*r*增加权重的值；同样，如果目标值为 0 且估计值为 1，我们希望减小权重，使得输入值不超过阈值。

1.  重复*步骤 3-4*，直到预测输出*y*和实际输出*yhat*之间的差值低于某个期望的阈值。在有非零偏置项*b*的情况下，也可以使用类似的公式来计算更新。

尽管简单，你可以理解这样的分类器可以学习到许多模式，但仍然不能学习到`XOR`函数。然而，通过将几个感知机组合成多个层，这些单元可以表示任何简单的布尔函数，¹⁰而且麦卡洛克和皮茨此前已经推测过将这些简单单元组合成一个通用计算引擎或图灵机，可以表示标准编程语言中的任何操作。然而，前述学习算法对每个单元独立操作，这意味着它可以扩展到由许多层感知机组成的网络（*图 3.3*）。

![](img/B16176_03_03.png)

图 3.3：一个多层感知机¹¹

然而，麻省理工学院的计算机科学家马文·明斯基和西摩·帕珀特在 1969 年的书籍《感知机》中表明，一个三层前馈网络需要至少有一个这些单元（在第一层）与所有输入之间的完全（非零权重）连接才能计算所有可能的逻辑输出¹²。这意味着，与生物神经元只连接到少数邻居相比，这些计算模型需要非常密集的连接。

虽然后来的架构中已经融入了连接的稀疏性，比如 CNNs，但这种密集的连接仍然是许多现代模型的特征，特别是在通常形成模型倒数第二层隐藏层的*全连接*层中。除了这些模型在当时的硬件上计算上不便利外，对于稀疏模型无法计算所有逻辑运算的观察被研究界更广泛地解释为*感知机无法计算 XOR*。虽然是错误的，¹³但这个观点导致了 AI 在随后的几年里资金的枯竭，有时这段时期被称为**AI 冬季**¹⁴。

神经网络研究的下一次革命将需要一种更有效的方法来计算复杂模型中更新所需的参数，这种技术将被称为**反向传播**。

# 多层感知机和反向传播

尽管自从《感知机》出版后，直到 1980 年代，神经网络的大规模研究资金都在下降，但研究人员仍然认识到这些模型有价值，特别是当它们被组装成由多个感知机单元组成的多层网络时。事实上，当输出函数的数学形式（即模型的输出）被放宽为多种形式（如线性函数或 Sigmoid 函数）时，这些网络可以解决回归和分类问题，理论结果表明，3 层网络可以有效逼近任何输出。¹⁵然而，这项工作没有解决这些模型的计算解的实际限制，而之前描述的感知机学习算法等规则对它们的应用造成了很大的限制。

对神经网络的重新关注始于反向传播算法的普及，该算法尽管在 20 世纪 60 年代已经被发现，但直到 20 世纪 80 年代才被广泛应用于神经网络，此前的多项研究强调了它在学习这些模型中的权重方面的有用性。¹⁶ 正如你在感知机模型中所看到的，更新权重的学习规则在没有“隐藏”层的情况下是相对容易推导出来的。输入只被感知机一次性地转换为输出值，意味着可以直接调整权重以产生期望的输出。当输入和输出之间有隐藏层时，问题就变得更加复杂：我们何时改变内部权重以计算输入权重遍历到最终输出的激活值？我们如何根据输入权重来修改它们？

反向传播技术的见解在于，我们可以利用微积分中的链式法则来高效地计算网络中每个参数相对于损失函数的导数，并且结合学习规则，这为训练多层网络提供了一种可扩展的方法。

让我们用一个例子来说明反向传播：考虑一个像*图 3.3*中所示的网络。假设最终层中的输出是使用 S 形函数计算的，这将产生一个值在 0 到 1 之间：

![](img/B16176_03_005.png)

此外，值*y*，即最终神经元的输入之和，是隐藏单元的 S 形输入的加权和：

![](img/B16176_03_006.png)

我们还需要一个概念，来判断网络在完成任务时是表现良好还是不良好。在这里可以使用的一个直观的误差函数是平方损失：

![](img/B16176_03_007.png)

其中*yhat*是估计值（来自模型输出）,*y*是所有输入示例*J*和网络*K*的输出的实际值的总和（其中*K=1*，因为只有一个输出值）。 反向传播开始于“前向传递”，在这一步中我们计算内层和外层所有输出的值，从而得到*yhat*的估计值。然后我们进行后向传递来计算梯度来更新权重。 

我们的总体目标是计算每个神经元的权重*w*和偏置项 b 的偏导数：![](img/B16176_03_008.png)和![](img/B16176_03_009.png)，这将使我们能够计算出*b*和*w*的更新。为了实现这个目标，让我们从计算最终神经元输入的更新规则开始；我们希望使用链式规则来计算误差*E*对于每个这些输入的偏导数（在本例中有五个，对应于五个隐藏层神经元）：

![](img/B16176_03_010.png)

我们可以通过对损失函数求导来得到值![](img/B16176_03_011.png)：

![](img/B16176_03_012.png)

对于单个示例，这只是输入和输出值之间的差异。对于![](img/B16176_03_013.png)，我们需要对 Sigmoid 函数进行偏导数：

![](img/B16176_03_014.png)![](img/B16176_03_015.png)![](img/B16176_03_016.png)

综上所述，我们有：

![](img/B16176_03_017.png)

如果我们想要计算特定参数*x*（如权重*w*或偏置项*b*）的梯度，我们需要多做一步：

![](img/B16176_03_018.png)

我们已经知道第一项，且*x*仅通过来自下层*y*的输入依赖于*w*，因为这是一个线性函数，所以我们得到：

![](img/B16176_03_019.png)

如果我们想为隐藏层中的一个神经元计算此导数，我们以同样的方式对这个输入*y*[i]进行偏导数计算，这很简单：

![](img/B16176_03_020.png)

因此，我们总共可以对所有输入到这个隐藏层的单元求和：

![](img/B16176_03_021.png)

我们可以递归地重复这个过程以获得所需的更新规则，因为我们现在知道如何在任何层计算*y*或*w*的梯度。这使得更新权重的过程变得高效，因为一旦我们通过反向传播计算了梯度，我们就可以结合连续的梯度通过层来得到网络任何深度所需的梯度。

现在，我们已经得到了每个*w*（或其他需要计算的神经元参数）的梯度，我们如何制定"学习规则"来更新权重？在他们的论文中，Hinton 等人指出，我们可以在每个样本批处理上计算梯度后应用更新，但建议在所有样本上计算平均值后应用更新。梯度表示误差函数相对于参数发生最大变化的方向；因此，为了更新，我们希望将权重推向*相反*的方向，*e*是一个小值（步长）：

![](img/B16176_03_023.png)

然后在训练过程中的每个时间点*t*，我们使用计算出的梯度更新权重：

![](img/B16176_03_024.png)

扩展这个方法，Hinton 等人提出了一个当前梯度的指数加权更新加上先前更新的方法：

![](img/B16176_03_025.png)

其中 alpha 是一个衰减参数，用于加权先前更新的贡献，取值范围从 0 到 1。根据这个过程，我们将使用一些小的随机值初始化网络中的权重，选择步长*e*，并通过前向和后向传播以及参数更新进行迭代，直到损失函数达到某个期望值。

现在我们已经描述了反向传播背后的形式数学，让我们看看它在实践中如何在 TensorFlow 2 等软件包中实现。

## 实践中的反向传播

虽然通过这种推导来理解深度神经网络的更新规则是有用的，但对于大型网络和复杂架构来说，这显然会很快变得难以管理。因此，幸运的是，TensorFlow 2 可以自动处理这些梯度的计算。在模型初始化期间，每个梯度都被计算为图中张量和操作之间的中间节点：例如，参见*图 3.4*：

![](img/B16176_03_04.png)

图 3.4：将梯度操作插入到 TensorFlow 图中¹⁸

在上述图的左侧显示了一个成本函数 *C*，它是从**修正线性单元**（**ReLU**）的输出中计算得到的（一种我们将在本章后面介绍的神经元函数），而这个输出又是通过将一个权重向量乘以输入 *x* 并添加一个偏置项 *b* 计算得到的。在右侧，你可以看到 TensorFlow 已经扩展了这个图，以计算作为整个控制流一部分所需的所有中间梯度。

在存储了这些中间值之后，通过递归操作将它们组合成完整的梯度的任务交给了 GradientTape API。在幕后，TensorFlow 使用一种称为**反向模式自动微分**的方法来计算梯度；它将依赖变量（输出*y*）固定，并且从网络的末端递归地向前计算所需的梯度。

例如，让我们考虑以下形式的神经网络：

![](img/B16176_03_05.png)

图 3.5：反向模式自动微分¹⁹

如果我们想要计算输出 *y* 关于输入 *x* 的导数，我们需要重复地代入最外层的表达式²⁰：

![](img/B16176_03_026.png)

因此，为了计算所需的梯度，我们只需从上到下遍历图，当我们计算时存储每个中间梯度。这些值被存储在一个记录上，被称为磁带，这是一个对早期计算机的参考，其中信息存储在磁带上，²¹然后用于重放值以进行计算。另一种方法是使用前向模式自动微分，从下到上计算。这需要两次而不是一次传递（对于每个馈入到最终值的分支），但在概念上更容易实现，不需要反向模式的存储内存。然而，更重要的是，反向模式模仿了我之前描述的反向传播的推导。

这个磁带（也称为**Wengert Tape**，以其开发者之一命名）实际上是一个数据结构，你可以在 TensorFlow Core API 中访问到。例如，导入核心库：

```py
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
```

然后，可以使用 `tf.GradientTape()` 方法来获取这个磁带，在其中你可以评估与图中间值相关的梯度²²：

```py
x = tf.ones((2, 2))
with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)
# Use the tape to compute the derivative of z with respect to the
# intermediate value y.
dz_dy = t.gradient(z, y)
# note that the resulting derivative, 2*y, = sum(x)*2 = 8
assert dz_dy.numpy() == 8.0 
```

默认情况下，`GradientTape()` 使用的内存资源在调用 `gradient()` 后被释放；但是，你也可以使用 `persistent` 参数来存储这些结果²³：

```py
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
  t.watch(x)
  y = x * x
  z = y * y
dz_dx = t.gradient(z, x)  # 108.0 (4*x³ at x = 3)
dy_dx = t.gradient(y, x)  # 6.0 
```

现在你已经看到 TensorFlow 如何实际计算梯度以评估反向传播，让我们回顾一下反向传播技术是如何随着时间的推移而发展，以应对实际实现中的挑战的细节。

## 反向传播的缺陷

虽然反向传播过程提供了一种以合理方式更新网络内部权重的方法，但它存在几个缺点，使得深度网络在实践中难以使用。其中一个是 **梯度消失** 的问题。在我们推导反向传播公式时，你看到网络中更深层次的权重的梯度是来自更高层的连续偏导数的乘积。在我们的例子中，我们使用了 Sigmoid 函数；如果我们绘制出 Sigmoid 的值及其一阶导数，我们可以看到一个潜在的问题：

![](img/B16176_03_06.png)

图 3.6：Sigmoid 函数及其梯度²⁴

随着 Sigmoid 函数的值向极端值（0 或 1，代表“关闭”或“打开”）增加或减少，梯度的值趋近于零。这意味着从隐藏激活函数 `y` 的这些梯度得到的更新值 `w` 和 `b` 会趋向于零，使得权重在迭代之间变化很小，使得反向传播过程中隐藏层神经元的参数变化非常缓慢。很显然，这里的一个问题是 Sigmoid 函数饱和；因此，选择另一个非线性函数可能会规避这个问题（这确实是作为 ReLU 提出的解决方案之一，我们稍后会讨论）。

另一个问题更微妙，与网络如何利用其可用的自由参数有关。正如你在 *第一章*，*生成型 AI 简介：“从模型中“绘制”数据* 中看到的，变量的后验概率可以计算为似然和先验分布的乘积。我们可以将深度神经网络看作是这种概率的图形表示：神经元的输出，取决于其参数，是所有输入值和这些输入上的分布（先验）的乘积。当这些值变得紧密耦合时就会出现问题。举个例子，考虑一下头痛的竞争性假设：

![](img/B16176_03_07.png)

图 3.7：解释逆效应

如果一个病人患有癌症，那么关于他们是否感冒的证据是如此压倒性，以至于没有提供额外价值；实际上，两个先前的假设的价值由于其中一个的影响而变得耦合。这使得计算不同参数的相对贡献变得棘手，特别是在深层网络中；我们将在我们关于《第四章，教网络生成数字》中讨论受限玻尔兹曼机和深度信念网络的问题。正如我们在该章节中将更详细地描述的那样，一项 2006 年的研究²⁵展示了如何抵消这种效应，这是对深度神经网络中可行推断的最早的一次突破，这一突破依赖于产生手绘数字图像的生成模型。

除了这些问题之外，在 20 世纪 90 年代和 21 世纪初，神经网络更广泛应用的其他挑战还包括像支持矢量机²⁶、梯度和随机梯度提升模型²⁷、随机森林²⁸甚至是惩罚回归方法如 LASSO²⁹和 Elastic Net³⁰这样的方法，用于分类和回归任务。

虽然理论上，深度神经网络的表征能力可能比这些模型更强，因为它们通过连续层构建输入数据的分层表示，与通过单一转换给出的“浅”表示如回归权重或决策树相反，但在实践中，训练深层网络的挑战使得这些“浅”方法对实际应用更有吸引力。这也与较大网络需要调整成千上万甚至是百万参数的事实相搭上了较大计算资源的事实，使这些实验在云供应商提供的廉价计算资源的爆炸之前是不可行的，包括 GPU 和 TPU 特别适用于快速矩阵计算。

现在我们已经介绍了训练简单网络架构的基础知识，让我们转向更复杂的模型，这些模型将构成书中许多生成模型的基础：CNNs 和序列模型（RNNs，LSTMs 等）。

# 网络的种类：卷积和递归

到目前为止，我们主要通过引用前馈网络来讨论神经网络的基础知识，其中每个输入都连接到每个层的每个输出。虽然这些前馈网络有助于说明深层网络的训练方式，但它们只是现代应用中使用的一类更广泛架构的一部分，包括生成模型。因此，在讨论使训练大型网络变得实用的一些技术之前，让我们回顾一下这些替代的深度模型。

## 视觉网络：卷积架构

正如本章开头所指出的，深度神经网络模型的灵感之一是生物神经系统。当研究人员试图设计可以模仿视觉系统功能的计算机视觉系统时，他们转向了视网膜的结构，这是在 20 世纪 60 年代神经生物学家 David Huber 和 Torsten Weisel 的生理学研究中揭示的。³¹ 正如以前所描述的，生理学家 Santiago Ramon Y Cajal 提供了神经结构如视网膜被安排在垂直网络中的视觉证据。

![](img/B16176_03_08_a+b.png)

图 3.8：视网膜的“深层神经网络”^(32 33)

Huber 和 Weisel 研究了猫的视网膜系统，展示了它们对形状的知觉是由排列在一列中的单个细胞的活动所组成的。每一列细胞都被设计用来检测输入图像中边缘的特定方向；复杂形状的图像是由这些简单图像拼接在一起的。

## 早期的 CNNs

这种列的概念启发了对 CNN 结构的早期研究³⁴。与前馈网络中学习单元之间的个体权重不同，这种结构（*图 3.9*）使用了专门用于检测图像中特定边缘的一组神经元中的共享权重。网络的初始层（标记为 **H1**）由每个 64 个神经元的 12 组组成。这些组中的每个都是通过在 16 x 16 像素的输入图像上传递一个 5 x 5 的网格来得到的；这个组中的每一个 64 个 5 x 5 的网格共享相同的权重，但与输入的不同空间区域相关联。你可以看到，如果它们的接受域重叠了两个像素，那么每个组中必须有 64 个神经元来覆盖输入图像。

当这 12 组神经元在 **H1** 层中结合在一起时，它们形成了 12 个表示图像中特定边缘的存在或不存在的 8 x 8 网格（*图 3.9*）。这种权重共享在直观上是有意义的，因为由权重表示的卷积核被指定用来检测图像中的不同颜色和/或形状，不管它出现在图像的哪个位置。这种降采样的效果是一定程度上的位置不变性；我们只知道边缘发生在图像某个区域内，但由于降采样导致的分辨率降低，我们无法知道确切位置。因为它们是通过将一个 5 x 5 的矩阵（卷积核）与图像的一部分相乘得到的，这种操作被用在图像模糊和其他转换中，这 5 x 5 的输入特征被称为 **卷积核**，也给网络起了名字。

![](img/B16176_03_09.png)

图 3.9：卷积神经网络³⁵

当我们有了这 12 个缩小了的 8 x 8 图像时，下一层（**H2**）还有 12 组神经元；在这里，卷积核是 5 x 5 x 8——它们横跨从**H1**上的一个 8 x 8 地图，遍及 12 个中的 8 个组。由于一个 5 x 5 的网格可以在 8 x 8 的网格上上下移动四次以覆盖 8 x 8 网格中的所有像素，我们需要 16 个这样的 5 x 5 x 8 组的神经元。

就像视觉皮层中更深层的细胞一样，网络中的更深层对来自不同边缘检测器的多个列进行整合，将信息组合在一起。

最后，该网络的第三个隐藏层（**H3**）包含 30 个隐藏单元和**H2**中的 12 x 16 个单元之间的全全连接，就像在传统的前馈网络中一样；最终的 10 个输出单元将输入图像分类为 10 个手写数字之一。

通过权重共享，在该网络中的自由参数总数得到了减少，虽然在绝对术语中仍然很大。虽然反向传播成功地用于此任务，但需要为一组成员受限的图像设计精心的网络，这些图像具有局限性的结果——对于如检测来自数百或数千个可能类别的对象等实际应用，需要采用其他方法。

## AlexNet 和其他 CNN 创新技术

2012 年的一篇文章产生了最先进的结果，使用一个被称为 AlexNet 的模型将 ImageNet 中的 130 万张图像分类为 1000 种分类。这些模型要实现训练，需要采用一些后来的创新技术。（36）如我之前提到的一样，一个是使用 ReLU（37）替代 sigmoid 或双曲正切函数。ReLU 是以下形式的函数：

![](img/B16176_03_027.png)

与 sigmoid 函数或 tanh 相比，在函数饱和时，其导数会缩小至 0，而 ReLU 函数具有恒定的梯度和 0 处的不连续性（*图 3.10*）。这意味着梯度不会饱和，导致网络的深层训练更慢，导致优化困难。

![](img/B16176_03_10.png)

图 3.10：替代激活函数的梯度（38）

虽然 ReLU 函数具有非消失梯度和低计算要求的优势（因为它们只是阈值线性变换），但缺点是如果输入低于 0，则它们可能会“关闭”，导致再次出现 0 梯度。这个问题在之后的工作中得到解决，在 0 以下引入了一个“泄漏”。（39）

![](img/B16176_03_028.png)

进一步的改进是使此阈值自适应，具有斜率为*a*的**参数化泄漏 ReLU**（**PReLU**）。（40）

![](img/B16176_03_029.png)

AlexNet 使用的另一个技巧是辍学。⁴¹ 辍学的想法受到合奏方法的启发，在合奏方法中，我们对许多模型的预测进行平均，以获得更稳健的结果。显然，对于深度神经网络来说，这是不可行的；因此，一个妥协方案是以 0.5 的概率随机将某些神经元的值设为 0。这些值在每次反向传播的前向传递中被重置，允许网络有效地对不同的架构进行采样，因为“辍学”的神经元在该传递中不参与输出。

![](img/B16176_03_11.png)

图 3.11：辍学

AlexNet 中使用的另一个增强是局部响应归一化。尽管 ReLU 不像其他单元那样饱和，模型的作者仍然发现限制输出范围有价值。例如，在一个单个卷积核中，他们使用相邻卷积核的值对输入进行归一化，这意味着总体响应被重新缩放⁴²：

![](img/B16176_03_030.png)

其中*a*是图像上给定*x*，*y*位置处的非标准化输出，*j*的总和是在相邻卷积核上，*B*，*k*和 alpha 是超参数。这种重新缩放让人想起后来被广泛应用于卷积和其他神经网络架构中的一种创新，批量归一化⁴³。批量归一化还对网络内部的“原始”激活应用转换：

![](img/B16176_03_031.png)

其中*x*是非标准化输出，*B*和*y*是尺度和偏移参数。这种转换被广泛应用于许多神经网络架构，以加速训练，尽管它的有效原因仍然是争论的话题。⁴⁴

现在你对使大型 CNN 训练成为可能的一些方法论进步有了一些了解，让我们来研究 AlexNet 的结构，看看我们将在后面章节中实现的生成模型中使用的一些额外的架构组件。

## AlexNet 架构

尽管*图 3.12*中的 AlexNet 架构看起来令人生畏，但一旦我们将这个大型模型分解为单独的处理步骤，就不那么难理解了。让我们从输入图像开始，跟踪通过每个后续神经网络层的一系列转换为每个图像计算输出分类的方法。

![](img/B16176_03_12.png)

图 3.12：AlexNet

输入到 AlexNet 的图像大小为 224 x 224 x 3（对于 RGB 通道）。第一层由 96 个单元和 11 x 11 x 3 卷积核组成；输出经过响应归一化（如前所述）和最大化池化。最大化池化是一种采取*n* x *n*网格上的最大值来记录输入中是否“任何位置”出现模式的操作；这又是一种位置不变性的形式。

第二层也是一组规模为 5 x 5 x 8 的卷积，以 256 个为一组。第三层到第五层都有额外的卷积，没有规范化，接着是两个全连接层和一个输出大小为 1,000 表示 ImageNet 中可能的图像类。AlexNet 的作者使用了几个 GPU 来训练模型，这种加速对输出非常重要。

![](img/B16176_03_13.png)

图 3.13：来自 AlexNet 的图像核

在初始的 11 x 11 x 3 卷积中，即训练过程中学到的特征中（*图 3.13*），我们可以看到可识别的边缘和颜色。虽然 AlexNet 的作者没有展示出网络中更高层次的神经元合成这些基本特征的例子，但另一项研究提供了一个示例，在该研究中，研究人员训练了一个大型的 CNN 来对 YouTube 视频中的图像进行分类，得到了网络最上层的一个神经元，它似乎是一个猫探测器（*图 3.14*）。

![](img/B16176_03_14.png)

图 3.14：从 YouTube 视频中学习到的猫探测器⁴⁵

这个概述应该让你明白 CNN 架构看起来的样子，以及什么样的发展使得它们随着时间的推移而成为图像分类器或基于图像的生成模型的基础更加可行。现在我们将转向另一类更专业的架构——RNN，这种架构用于开发时间或基于序列的模型。

# 序列数据的网络

除了图像数据，自然语言文本也一直是神经网络研究中的一个热门话题。然而，与我们迄今为止检查的数据集不同，语言有一个重要的*顺序*与其含义相关。因此，为了准确地捕捉语言或时间相关数据中的模式，有必要使用专门设计用于此目的的网络。

## RNN 和 LSTM

让我们想象一下，我们试图预测句子中的下一个词，给定到目前为止的词。试图预测下一个词的神经网络不仅需要考虑当前词，还需要考虑可变数量的先前输入。如果我们只使用一个简单的前馈 MLP，该网络实际上会将整个句子或每个词都处理为一个向量。这引入了这样一个问题：要么必须将可变长度的输入填充到一个共同的长度，并且不保留任何相关性的概念（也就是说，在生成下一个预测时，句子中哪些单词比其他单词更相关），或者在每一步中只使用上一个词作为输入，这样会丢失句子其余部分的上下文和提供的所有信息。这种问题激发了“原生”RNN⁴⁶，它在计算一个神经元的输出时，不仅考虑当前输入，还考虑前一步的隐藏状态：

![](img/B16176_03_032.png)

可以将这个过程想象为每一层递归地馈送到下一个时间步骤的序列中。实际上，如果我们“展开”序列的每个部分，我们最终得到一个非常深的神经网络，其中每一层共享相同的权重。⁴⁷

![图片](img/B16176_03_15.png)

图 3.15：展开的 RNN⁴⁸

训练深度前馈网络所具有的困难也同样适用于循环神经网络；使用传统激活函数时，梯度往往在长距离上衰减（或者如果梯度大于 1，则爆炸）。

然而，与前馈网络不同，RNNs 不是用传统的反向传播进行训练，而是用一种称为**时间反向传播**（**BPTT**）的变体：网络被展开，如前所述，使用反向传播，对每个时间点的误差进行平均处理（因为每一步都有一个“输出”，即隐藏状态）。⁴⁹此外，在 RNNs 的情况下，我们遇到的问题是网络的记忆非常短暂；它只包含最近单元的信息，而当前单元之前的信息则难以保持长期上下文。对于翻译等应用来说，这显然是一个问题，因为句子末尾的单词的解释可能依赖于句子开头的术语，而不仅仅是直接前面的术语。

LSTM 网络的开发是为了使 RNNs 能够在长序列上保持上下文或状态。⁵⁰

![图片](img/B16176_03_16.png)

图 3.16：LSTM 网络

在传统的 RNN 中，我们只保留来自前一步隐藏单元激活的短期记忆*h*。除了这个短期记忆外，LSTM 架构引入了一个额外的层*c*，即“长期”记忆，它可以持续多个时间步长。从某种意义上说，这种设计让人想起了电容器，它可以使用*c*层来储存或保持“电荷”，一旦达到某个阈值就释放它。为了计算这些更新，一个 LSTM 单元由许多相关的神经元或门组成，这些门在每个时间步骤上一起作用来转换输入。

给定输入向量*x*和前一时刻*t-1*的隐藏状态*h*，在每个时间步长，LSTM 首先计算了一个值，从 0 到 1 表示*c*的每个元素中“遗忘”了多少信息：

![图片](img/B16176_03_033.png)

我们进行第二次类似的计算来确定要保留输入值的哪些部分：

![图片](img/B16176_03_034.png)

现在我们知道了*c*的哪些元素被更新了；我们可以计算这个更新如下：

![图片](img/B16176_03_035.png)

其中![图片](img/B16176_03_036.png)是一个 Hadamard 积（逐元素乘法）。本质上，这个方程告诉我们如何使用 tanh 变换计算更新，使用输入门过滤它们，并使用忘记门将它们与前一个时间步的长期记忆结合起来，以潜在地过滤掉旧值。

要计算每个时间步的输出，我们计算另一个输出门：

![图片](img/B16176_03_037.png)

并且在每一步计算最终输出时（隐藏层作为下一步的短期记忆提供给下一步），我们有：

![](img/B16176_03_038.png)

提出了许多这种基本设计的变体；例如，“窥视孔”LSTM 用*c*(*t*-*1*)替代了*h*(*t*-*1*)（因此每个操作都可以“窥视”长期记忆单元），⁵¹而 GRU⁵²通过删除输出门简化了整体设计。这些设计的共同之处在于，它们避免了训练 RNN 时出现的梯度消失（或爆炸）困难，因为长期记忆充当缓冲区，以维持梯度并在许多时间步骤上传播神经元激活。

# 构建更好的优化器

到目前为止，在本章中，我们已经讨论了几个例子，其中更好的神经网络架构实现了突破；然而，与此同样（甚至更加）重要的是用于在这些问题中最小化误差函数的*优化过程*，通过选择产生最低误差的参数来“学习”网络的参数。回顾我们对反向传播的讨论，这个问题有两个组成部分：

+   **如何初始化权重**：在许多历史应用中，我们看到作者使用了一定范围内的随机权重，并希望通过反向传播的使用从这个随机起始点至少得到一个局部最小化的损失函数。

+   **如何找到局部最小损失**：在基本的反向传播中，我们使用梯度下降和固定学习率以及一阶导数更新来遍历权重矩阵的潜在解空间；然而，有充分的理由相信可能存在更有效的方法来找到局部最小值。

事实上，这两者都被证明是深度学习研究进展的关键考虑因素。

## 梯度下降到 ADAM

正如我们在反向传播的讨论中看到的那样，1986 年提出的用于训练神经网络的原始版本在获取梯度并更新权重之前对*整个数据集*进行了损失平均。显然，这相当慢，并且使模型的分发变得困难，因为我们无法分割输入数据和模型副本；如果我们使用它们，每个副本都需要访问整个数据集。

相比之下，SGD 在*n*个样本后计算梯度更新，其中*n*可以是从 1 到*N*（数据集的大小）的范围。在实践中，我们通常执行*小批量*梯度下降，其中*n*相对较小，而且我们在每个 epoch（数据的一次遍历）后随机分配数据给*n*批次。

但是，SGD 可能会很慢，导致研究人员提出加速搜索最小值的替代方案。正如在原始反向传播算法中所见，一个想法是使用一种记住先前步骤并在前进方向继续的指数加权动量形式。已经有提出了各种变体，如*Nesterov Momentum*，它增加了一个项来增加这种加速^（53）。

![](img/B16176_03_039.png)![](img/B16176_03_040.png)

与原始反向传播算法中使用的动量项相比，将当前动量项加到梯度中有助于保持动量部分与梯度变化保持一致。

另一种优化方法，称为**自适应梯度**（**Adagrad**）^（54），通过该参数梯度的平方和（*G*）来缩放每次更新的学习率；因此，经常更新的元素被降采样，而不经常更新的元素被推动以更大的幅度进行更新：

![](img/B16176_03_041.png)

这种方法的缺点是，随着我们继续训练神经网络，总和*G*将无限增加，最终将学习率缩小到一个非常小的值。为了解决这个缺点，提出了两种变体方法，RMSProp^（55）（经常应用于 RNN）和 AdaDelta^（56），在计算*G*时加入固定宽度窗口的 n 步。

**自适应动量估计**（**ADAM**）^（57）可以看作是一种尝试将动量和 AdaDelta 结合起来；动量计算用于保留过去梯度更新的历史，而在 AdaDelta 中使用的固定更新窗口内的衰减平方梯度总和用于调整结果梯度的大小。

这里提到的方法都具有*一阶*的特性：它们只涉及损失对输入的一阶导数。虽然计算简单，但这可能导致在神经网络参数的复杂解空间中导航时出现实际挑战。如*图 3.17*所示，如果我们将权重参数的景观视为一条沟壑，那么一阶方法要么在曲率快速变化的区域移动得太快（顶部图像），超调极小值，要么在曲率较低的极小值“沟壑”中移动得太慢。理想的算法将考虑曲率和曲率变化的*变化速率*，允许优化器顺序方法在曲率变化特别缓慢时采用更大的步长，反之亦然（底部图像）。

![](img/B16176_03_17.png)

图 3.17：复杂的景观和二阶方法^（58）

因为它们利用了导数的改变速率（**二阶导数**），这些方法被称为**二阶**，并且在优化神经网络模型中已经取得了一定的成功^（59）。

然而，每次更新所需的计算量比一阶方法大，因为大多数二阶方法涉及大型矩阵求逆（因此内存利用率高），需要近似来使这些方法可扩展。然而，最终，实际优化网络的突破之一不仅来自于优化算法，还包括我们如何初始化模型中的权重。

## Xavier 初始化

正如之前所述，在早期研究中，常常用一定范围的随机值初始化神经网络的权重。2006 年在深度置信网络的训练中取得的突破，正如您将在*第四章*，*教授网络生成数字*中看到的那样，使用了预训练（通过生成建模方法）来在执行标准反向传播之前初始化权重。

如果您曾经在 TensorFlow Keras 模块中使用过一个层，您会注意到层权重的默认初始化是从截断的正态分布或均匀分布中抽取的。这个选择是从哪里来的？正如我之前描述的，使用 S 型或双曲线激活函数的深度网络的一个挑战是，它们倾向于变得饱和，因为这些函数的值受到非常大或负的输入的限制。我们可以解释初始化网络的挑战是保持权重在这样一个范围内，以至于它们不会使神经元的输出饱和。另一种理解方法是假设神经元的输入和输出值具有类似的方差；信号在通过神经元时不会被大幅放大或减小。

在实践中，对于一个线性神经元，*y* = *wx* + *b*，我们可以计算输入和输出的方差为：

![](img/B16176_03_042.png)

*b*是常数，因此我们剩下：

![](img/B16176_03_043.png)

由于权重矩阵中有*N*个元素，并且我们希望*var*(*y*)等于*var*(*x*)，这给出了：

![](img/B16176_03_044.png)

因此，对于权重矩阵*w*，我们可以使用方差为 1/*N*（输入和输出单元的平均数量，因此权重的数量）的截断正态分布或均匀分布。⁶⁰变体也已经应用于 ReLU 单元：⁶¹这些方法被称为它们原始作者的名字，如 Xavier 或 He 初始化。

总的来说，我们回顾了 TensorFlow 2 中底层使用的几种常见优化器，并讨论了它们如何改进基本的 SGD 形式。我们还讨论了聪明的权重初始化方案如何与这些优化器共同作用，使我们能够训练越来越复杂的模型。

# 摘要

在本章中，我们涵盖了深度学习的基本词汇 - 如起始对感知器和多层感知器的研究导致了简单的学习规则被放弃，而采用反向传播。我们还研究了专门的神经网络架构，如基于视觉皮层的卷积神经网络（CNNs），以及专门用于序列建模的循环网络。最后，我们检查了最初为反向传播提出的梯度下降算法的变体，这些变体的优点包括*动量*，并描述了将网络参数放在更容易导航到局部最小值范围的权重初始化方案。

在这种背景下，我们将着手进行生成模型的项目，首先是使用深度信念网络生成 MNIST 数字的项目，见*第四章*，*教授网络生成数字*。

# 参考文献

1.  López-Muñoz F., Boya J., Alamo C. (2006). *神经元理论，神经科学的基石，颁给圣地亚哥·拉蒙·伊·卡哈尔的诺贝尔奖 100 周年*。《大脑研究公报》. 70 （4–6）：391–405\. [`pubmed.ncbi.nlm.nih.gov/17027775/`](https://pubmed.ncbi.nlm.nih.gov/17027775/)

1.  Ramón y Cajal, Santiago (1888). *鸟类中枢神经中枢结构*。

1.  McCulloch, W.S., Pitts, W. (1943). *神经活动中所固有的思想的逻辑演算。数理生物物理学通报*5, 115–133\. [`doi.org/10.1007/BF02478259`](https://doi.org/10.1007/BF02478259)

1.  请注意：Rashwan M., Ez R., reheem G. (2017). *阿拉伯语言语音识别的计算智能算法*.《开罗大学工程领域杂志》. 12\. 886-893\. 10.21608/auej.2017.19198\. [`wwwold.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node12.html`](http://wwwold.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node12.html)

1.  Rashwan M., Ez R., reheem G. (2017). *阿拉伯语言语音识别的计算智能算法*.《开罗大学工程领域杂志》. 12\. 886-893\. 10.21608/auej.2017.19198\. [`wwwold.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node12.html`](http://wwwold.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node12.html)

1.  *人工神经元*. 维基百科. 检索日期：2021 年 4 月 26 日，网址：[`en.wikipedia.org/wiki/Artificial_neuron`](https://en.wikipedia.org/wiki/Artificial_neuron)

1.  Shackleton-Jones Nick. (2019 年 5 月 3 日).*人们如何学习：设计教育和培训，以提高绩效*。Kogan Page。英国伦敦

1.  Hebb, D. O. (1949). *行为组织：神经心理学理论*。纽约：Wiley 和 Sons 出版社

1.  Rosenblatt, Frank (1957). *感知器-一个认知和识别自动装置*。报告 85-460-1. 康奈尔航空实验室。

1.  Marvin Minsky 和 Seymour Papert，1972 年（第二版，第一版 1969 年）《感知器：计算几何的介绍》，MIT 出版社，剑桥，马萨诸塞州

1.  Hassan, Hassan & Negm, Abdelazim & Zahran, Mohamed & Saavedra, Oliver. (2015). *利用高分辨率卫星图像评估人工神经网络进行浅水湖泊水深估计：以 El Burullus Lake 为例*. 国际水技术期刊. 5.

1.  Marvin Minsky 和 Seymour Papert, 1972 (第二版带有更正，第一版 1969) *感知机：计算几何简介*, The MIT Press, 剑桥 MA

1.  Pollack, J. B. (1989). "无意伤害：感知机扩展版评论". *数学心理学杂志*. 33 (3): 358–365.

1.  Crevier, Daniel (1993), *AI：人工智能的动荡探索*, 纽约，纽约: BasicBooks.

1.  Cybenko, G. *通过 S 型函数的叠加进行逼近*. 数学. 控制信号系统 2, 303–314 (1989). [`doi.org/10.1007/BF02551274`](https://doi.org/10.1007/BF02551274)

1.  Goodfellow, Ian; Bengio, Yoshua; Courville, Aaron (2016). *6.5 反向传播和其他差分算法*. 深度学习. MIT 出版社. pp. 200–220

1.  Rumelhart, D., Hinton, G. & Williams, R. (1986) *通过反向传播误差学习表示*. *自然* 323, 533–536\. [`doi.org/10.1038/323533a0`](https://doi.org/10.1038/323533a0)

1.  Guess A R., (2015 年 11 月 10 日). *Google 开源机器学习库 TensorFlow*. DATAVERSITY. [`www.dataversity.net/google-open-sources-machine-learning-library-tensorflow/`](https://www.dataversity.net/google-open-sources-machine-learning-library-tensorflow/)

1.  Berland (2007). *ReverseaccumulationAD.png*. 维基百科. 可从: [`commons.wikimedia.org/wiki/File:ReverseaccumulationAD.png`](https://commons.wikimedia.org/wiki/File:ReverseaccumulationAD.png)

1.  *自动微分*. 维基百科. [`en.wikipedia.org/wiki/Automatic_differentiation`](https://en.wikipedia.org/wiki/Automatic_differentiation)

1.  R.E. Wengert (1964). *一个简单的自动导数评估程序*. Comm. ACM. 7 (8): 463–464.;Bartholomew-Biggs, Michael; Brown, Steven; Christianson, Bruce; Dixon, Laurence (2000). *算法的自动微分*. 计算与应用数学杂志. 124 (1–2): 171–190.

1.  TensorFlow 作者 (2018). *automatic_differentiation.ipynb*. 可从: [`colab.research.google.com/github/tensorflow/tensorflow/blob/r1.9/tensorflow/contrib/eager/python/examples/notebooks/automatic_differentiation.ipynb#scrollTo=t09eeeR5prIJ`](https://colab.research.google.com/github/tensorflow/tensorflow/blob/r1.9/tensorflow/contrib/eager/py)

1.  TensorFlow 作者. *梯度和自动微分简介*. TensorFlow. 可从: [`www.tensorflow.org/guide/autodiff`](https://www.tensorflow.org/guide/autodiff)

1.  Thomas (2018). *梯度消失问题和 ReLU – TensorFlow 调查*. 机器学习冒险。查阅：[`adventuresinmachinelearning.com/vanishing-gradient-problem-tensorflow/`](https://adventuresinmachinelearning.com/vanishing-gradient-problem-tensorflow/)

1.  Hinton, Osindero, Yee-Whye (2005). *深度信念网络的快速学习算法*. 多伦多大学，计算机科学。查阅：[`www.cs.toronto.edu/~fritz/absps/ncfast.pdf`](http://www.cs.toronto.edu/~fritz/absps/ncfast.pdf)

1.  Cortes, C., Vapnik, V. *支持向量网络*. 机器学习 20, 273–297 (1995). [`doi.org/10.1007/BF00994018`](https://doi.org/10.1007/BF00994018)

1.  Friedman, J. H. (February 1999). *贪婪函数逼近：梯度增强机* (PDF)

1.  Breiman, L. *随机森林*. 机器学习 45, 5–32 (2001). [`doi.org/10.1023/A:1010933404324`](https://doi.org/10.1023/A:1010933404324)

1.  Tibshirani R. (1996). *通过套索实现回归收缩和选择*. 英国皇家统计学会杂志。Wiley. 58 (1): 267–88.

1.  Zou H., Hastie T. (2005). *通过弹性网络实现正规化和变量选择*. 英国皇家统计学会杂志 B 系列：301–320

1.  Hubel D. H., Wiesel T. N. (1962) *感觉野，视交互及猫脑视觉皮层功能体系结构*. 生理学杂志，1962, 160: 106-154。[`doi.org/10.1113/jphysiol.1962.sp006837`](https://doi.org/10.1113/jphysiol.1962.sp006837)

1.  [`charlesfrye.github.io/FoundationalNeuroscience/img/corticalLayers.gif`](http://charlesfrye.github.io/FoundationalNeuroscience/img/corticalLayers.gif)

1.  Wolfe, Kluender, Levy (2009). *感知和知觉*. 坎伯兰：Sinauer Associates Inc.。

1.  LeCun, Yann, et al. *反向传播应用于手写邮政编码识别*. 神经计算，1.4 (1989): 541-551.

1.  LeCun, Yann, et al. *反向传播应用于手写邮政编码识别*. 神经计算，1.4 (1989): 541-551.

1.  *使用深度卷积神经网络进行 ImageNet 分类*：[`www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf`](https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convoluti)

1.  Nair V., Hinton G E. (2010). *修正线性单元改进限制玻尔兹曼机*. 机器学习国际会议论文集，2010 年，以色列海法。

1.  Agarap A F. (2019, September 5). *通过梯度噪音添加来避免伴随梯度消失的问题*. 朝着数据科学。[`towardsdatascience.com/avoiding-the-vanishing-gradients-problem-96183fd03343`](https://towardsdatascience.com/avoiding-the-vanishing-gradients-problem-96183fd03343)

1.  Maas A L., Hannun A Y., Ng A Y. (2013). *修正线性非线性改进神经网络声学模型*. 机器学习国际会议论文集，2013 年，美国佐治亚州亚特兰大市。

1.  He，K.，Zhang，X.，Ren，S.，Sun，J.（2015）。 *深入挖掘整流器：在 ImageNet 分类上超越人类水平性能*。 arXiv：1502.01852。[`arxiv.org/abs/1502.01852`](https://arxiv.org/abs/1502.01852)

1.  Hinton，G E.，Srivastava，N.，Krizhevsky，A.，Sutskever，I.，Salakhutdinov，R R.（2012）。 *通过防止特征检测器的协同适应来改进神经网络*。 arXiv：1207.0580。[`arxiv.org/abs/1207.0580`](https://arxiv.org/abs/1207.0580)

1.  Krizhevsky A.，Sutskever I.，Hinton G E.（2012）。 *使用深度卷积神经网络的 ImageNet 分类*。神经信息处理系统 25（NIPS 2012）的一部分。[`papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf`](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

1.  Ioffe，S.，Szegedy，C.（2015）。 *批量归一化：通过减少内部协变量转移加速深层网络训练*。 arXiv：1502.03167。 [`arxiv.org/abs/1502.03167`](https://arxiv.org/abs/1502.03167)

1.  Santurkar，S。，Tsipras，D。，Ilyas，A。，Madry，A.（2019）。 *批量归一化如何帮助优化？* arXiv：1805.11604。 [`arxiv.org/abs/1805.11604`](https://arxiv.org/abs/1805.11604)

1.  Dean J.，Ng，A Y.（2012）。 *使用大规模脑模拟进行机器学习和人工智能*。The Keyword | Google。[`blog.google/technology/ai/using-large-scale-brain-simulations-for/`](https://blog.google/technology/ai/using-large-scale-brain-simulations-for/)

1.  Rumelhart，D.，Hinton，G.和 Williams，R.（1986 年）*通过反向传播错误学习表示*。 *自然* 323，533–536。 [`doi.org/10.1038/323533a0`](https://doi.org/10.1038/323533a0)

1.  LeCun，Y.，Bengio，Y.和 Hinton G.（2015）。 *深度学习*。 *自然* 521，436–444。 [`www.nature.com/articles/nature14539.epdf`](https://www.nature.com/articles/nature14539.epdf)

1.  Olah（2015 年）。 *理解 LSTM 网络*. colah 的博客。可从[`colah.github.io/posts/2015-08-Understanding-LSTMs/`](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)获取。

1.  Mozer，M. C.（1995）。 *用于时间模式识别的聚焦反向传播算法*。在 Chauvin，Y .; Rumelhart，D.（eds。）。 *反向传播：理论，体系结构和应用*。 ResearchGate。 Hillsdale，NJ：劳伦斯 Erlbaum 凯斯。第 137-169 页。

1.  Greff K.，Srivastava，R K。，Koutník，J.，Steunebrink，B R。，Schmidhuber，J.（2017）。 *LSTM：搜索空间奥德赛*。 arXiv：1503.04069v2。 [`arxiv.org/abs/1503.04069v2`](https://arxiv.org/abs/1503.04069v2)

1.  Gers FA, Schmidhuber E. *LSTM 循环网络学习简单的无上下文和有上下文的语言*. IEEE 交易神经网络。 2001 年;12（6）：1333-40. doi：10.1109/72.963769。 PMID：18249962。

1.  Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., Bengio, Y. (2014). *使用 RNN 编码器-解码器学习短语表示用于统计机器翻译*。arXiv:1406.1078。[`arxiv.org/abs/1406.1078`](https://arxiv.org/abs/1406.1078)

1.  Sutskever, I., Martens, J., Dahl, G. & Hinton, G. (2013). *初始化和动量在深度学习中的重要性*。第 30 届国际机器学习大会论文集, PMLR 28(3):1139-1147.

1.  Duchi J., Hazan E., Singer Y. (2011). *用于在线学习和随机优化的自适应次梯度方法*。机器学习研究杂志 12 (2011) 2121-2159.

1.  Hinton, Srivastava, Swersky. *神经网络用于机器学习*，第 6a 讲。可从：[`www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf`](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

1.  Zeiler, M D. (2012). *ADADELTA：一种自适应学习率方法*。arXiv:1212.5701。[`arxiv.org/abs/1212.5701`](https://arxiv.org/abs/1212.5701)

1.  Kingma, D P., Ba, J. (2017). *Adam：一种随机优化方法*。arXiv:1412.6980。[`arxiv.org/abs/1412.6980`](https://arxiv.org/abs/1412.6980)

1.  Martens J. (2010). *通过无 Hessian 优化的深度学习*。ICML. Vol. 27. 2010.

1.  Martens J. (2010). *通过无 Hessian 优化的深度学习*。ICML. Vol. 27. 2010.

1.  Glorot X., Bengio Y., (2010). *理解训练深度前馈神经网络的困难*。第十三届人工智能与统计国际会议论文集。

1.  He, K., Zhang, X., Ren, S., Sun, J. (2015). *深入研究整流器：在 ImageNet 分类上超越人类水平性能*。arXiv:1502.01852。[`arxiv.org/abs/1502.01852`](https://arxiv.org/abs/1502.01852)