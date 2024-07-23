# 梯度下降及其变体

梯度下降是最流行和广泛使用的优化算法之一，是一种一阶优化算法。一阶优化意味着我们只计算一阶导数。正如我们在第一章中看到的，*深度学习简介*，我们使用梯度下降计算损失函数相对于网络权重的一阶导数以最小化损失。

梯度下降不仅适用于神经网络，它还用于需要找到函数最小值的情况。在本章中，我们将深入探讨梯度下降，从基础知识开始，并学习几种梯度下降算法的变体。有各种各样的梯度下降方法用于训练神经网络。首先，我们将了解**随机梯度下降**（**SGD**）和小批量梯度下降。然后，我们将探讨如何通过动量加速梯度下降以达到收敛。本章后期，我们将学习如何使用各种算法（如 Adagrad、Adadelta、RMSProp、Adam、Adamax、AMSGrad 和 Nadam）以自适应方式执行梯度下降。我们将使用简单的线性回归方程，并看看如何使用各种梯度下降算法找到线性回归成本函数的最小值。

在本章中，我们将学习以下主题：

+   解析梯度下降

+   梯度下降与随机梯度下降的区别

+   动量和 Nesterov 加速梯度

+   梯度下降的自适应方法

# 解析梯度下降

在深入细节之前，让我们先了解基础知识。数学中的函数是什么？函数表示输入和输出之间的关系。我们通常用![](img/340fb1cd-80df-4c80-9d3e-cf73319d7dec.png)表示一个函数。例如，![](img/0fae6909-055b-4274-a8b0-10c208a48914.png) 表示一个以![](img/ec3068c5-68a5-4d37-8497-2c7198921ff6.png) 为输入并返回![](img/bcec9718-a32d-4cbb-8bd2-be81646ed456.png) 为输出的函数。它也可以表示为![](img/d9a2a1f7-670d-459c-8d69-2bbf1bda9919.png)。

这里，我们有一个函数，![](img/f858029e-d5a6-42b2-adb1-08c259e11ed5.png)，我们可以绘制并查看函数的形状：

![](img/713266f0-94bb-470f-aede-dca33212cf32.png)

函数的最小值称为**函数的最小值**。正如您在上图中看到的，![](img/a3b99c72-2a57-45df-804e-1b5a5a245e50.png) 函数的最小值位于 0 处。前述函数称为**凸函数**，在这种函数中只有一个最小值。当存在多个最小值时，函数称为**非凸函数**。如下图所示，非凸函数可以有许多局部最小值和一个全局最小值，而凸函数只有一个全局最小值：

![](img/a80ce143-5782-4b43-bdad-98ed35b8d159.png)

通过观察 ![](img/fc61e336-b53b-4577-b381-3270a8b3d136.png) 函数的图表，我们可以很容易地说它在 ![](img/907d9105-3f08-40a8-948f-b85295a9d4db.png) 处有其最小值。但是，如何在数学上找到函数的最小值？首先，让我们假设 *x = 0.7*。因此，我们处于一个 *x = 0.7* 的位置，如下图所示：

![](img/cf53ffce-63b6-4901-a23d-29c3f99031a8.png)

现在，我们需要走向零，这是我们的最小值，但我们如何达到它？我们可以通过计算函数的导数，![](img/9004f0b2-5eb0-428a-85ad-65d0ff49998b.png)，来达到它。因此，关于 ![](img/ff9989d9-4558-42e3-bd59-73106825b532.png)，函数的导数为：

![](img/f0d41b34-b742-47eb-9e2c-41704d1c8f4c.png)

![](img/5685c5da-2d00-4003-9e4a-0bc98bb1cc83.png)

由于我们在 *x = 0.7* 处，并将其代入前述方程，我们得到以下方程：

![](img/e909932d-23a8-4f59-bff5-e166f37f083d.png)

计算导数后，我们根据以下更新规则更新 ![](img/82b7a97b-5bfc-4af7-acec-9a68bcf9d2ca.png) 的位置：

![](img/fe5a13ba-b2b8-4c5f-9ecb-ee86b960dba0.png)

![](img/5c84f501-8cf9-4a1d-963d-a65c840c8165.png)

![](img/b27d8d74-0525-466b-840f-0789c16f516e.png)

如我们在下图中所见，我们最初位于 *x = 0.7* 处，但在计算梯度后，我们现在处于更新位置 *x = -0.7*。然而，这并不是我们想要的，因为我们错过了我们的最小值 *x = 0*，而是达到了其他位置：

![](img/88c482a0-2098-4adb-95cc-953ca861c6ad.png)

为了避免这种情况，我们引入了一个称为学习率的新参数，![](img/2a4a3822-b1b0-4520-8ef5-350a15b5f201.png)，在更新规则中。它帮助我们减慢梯度下降的步伐，以便我们不会错过最小点。我们将梯度乘以学习率，并更新 ![](img/ed14be6e-8dfd-4224-8830-90929ef2bbf3.png) 的值，如下所示：

![](img/6d9aa0e7-19cc-4a64-9c4a-fbe7c95975c9.png)

假设 ![](img/70c9a5ba-c40e-4ca5-89f7-268009b00abb.png)；现在，我们可以写出以下内容：

![](img/dd07f2c3-8d87-47c1-8f28-9681313481dd.png)

![](img/60110808-4e9c-4040-8c90-16e12b6ea20f.png)

正如我们在下图中看到的那样，在将更新后的 *x* 值的梯度乘以学习率后，我们从初始位置 *x = 0.7* 下降到 *x = 0.49*：

![](img/a29a1581-00d6-465b-86cc-512afe8ee727.png)

然而，这仍然不是我们的最优最小值。我们需要进一步下降，直到达到最小值，即 *x = 0*。因此，对于一些 *n* 次迭代，我们必须重复相同的过程，直到达到最小点。也就是说，对于一些 *n* 次迭代，我们使用以下更新规则更新 *x* 的值，直到达到最小点：

![](img/0ab193ae-0792-4fa9-9fea-4898400c84ee.png)

好的，为什么在前面的方程中有一个减号？也就是说，为什么我们要从*x*中减去![](img/805ab6e1-eaf0-49d0-9b00-8a0cfeac074f.png)？为什么不能将它们加起来，将我们的方程变为![](img/21191f6f-8453-4570-ad5e-8db748a0fd65.png)？

这是因为我们在寻找函数的最小值，所以我们需要向下。如果我们将*x*加上![](img/ce311d68-a009-4bb9-b9f7-f2f1eaa930d9.png)，那么我们在每次迭代时都会向上移动，无法找到最小值，如下图所示：

| ![](img/75ed568d-5953-4ce8-a6c2-9734dd515c26.png) | ![](img/590de2fe-7d7c-4dab-b585-19e1ec5cff25.png) |
| --- | --- |
| ![](img/3cdb85c9-c1d6-4893-8216-b8456584ee25.png) | ![](img/f4945d58-7c63-4fcf-a3a3-c19632de3d3a.png) |

因此，在每次迭代中，我们计算*y*相对于*x*的梯度，即![](img/4494ff4f-e578-4cb8-bb05-570ca8f80c2d.png)，将梯度乘以学习率，即![](img/81159fea-9eb1-4e14-9b44-8a4d6364db92.png)，然后从*x*值中减去它以获得更新后的*x*值，如下所示：

![](img/f1573f8a-e46d-4af8-b06c-d0235a921102.png)

通过在每次迭代中重复此步骤，我们从成本函数向下移动，并达到最小点。正如我们在下图中所看到的，我们从初始位置 0.7 向下移动到 0.49，然后从那里到达 0.2。

然后，在多次迭代之后，我们达到了最小点，即 0.0：

![](img/ef6d023b-aef5-4fd5-ade2-c6cd0fc63fc7.png)

当我们达到函数的最小值时，我们称之为**收敛**。但问题是：我们如何知道我们已经达到了收敛？在我们的例子中，![](img/0bc8c52b-d12f-4341-ae71-a5d95cf472ce.png)，我们知道最小值是 0。因此，当我们达到 0 时，我们可以说我们找到了最小值，即我们已经达到了收敛。但是我们如何在数学上表达 0 是函数![](img/6aecabbf-04a4-4771-89af-4d9332c7ed27.png)的最小值呢？

让我们仔细观察下面的图表，显示了*x*在每次迭代中的变化。正如您可能注意到的那样，第五次迭代时*x*的值为 0.009，第六次迭代时为 0.008，第七次迭代时为 0.007。正如您所见，第五、六和七次迭代之间几乎没有什么差异。当*x*在迭代中的值变化很小时，我们可以得出我们已经达到了收敛：

![](img/85e06a85-c2e2-46b9-863b-db9f71f33379.png)

好了，但这一切有什么用？我们为什么要找到一个函数的最小值？当我们训练模型时，我们的目标是最小化模型的损失函数。因此，通过梯度下降，我们可以找到成本函数的最小值。找到成本函数的最小值给我们提供了模型的最优参数，从而可以获得最小的损失。一般来说，我们用![](img/870699d5-07f1-45c0-9e0c-522b85807918.png)来表示模型的参数。以下方程称为参数更新规则或权重更新规则：

![](img/85b286b3-5625-4888-b6c2-c01b4b5ed3d4.png)

在这里，我们有以下内容：

+   ![](img/7c47bd99-cc4d-4d46-af9a-6d91ce95c3ac.png)是模型的参数

+   ![](img/63d79fb5-00d3-4d7c-8959-770bb6f73a5a.png)是学习率

+   ![](img/2d536939-b2e6-44c5-9ba4-5e87c2b4e5ec.png)是梯度

我们根据参数更新规则多次迭代更新模型的参数，直到达到收敛。

# 在回归中执行梯度下降

到目前为止，我们已经理解了梯度下降算法如何找到模型的最优参数。在本节中，我们将了解如何在线性回归中使用梯度下降，并找到最优参数。

简单线性回归的方程可以表达如下：

![](img/97c861f0-d573-433a-b334-83319c22e8e2.png)

因此，我们有两个参数，![](img/949a79f2-a8f5-4c59-97b3-81e5a3e0fc9d.png)和![](img/1134aef2-52c6-496a-972f-0db902f14e5f.png)。现在，我们将看看如何使用梯度下降找到这两个参数的最优值。

# 导入库

首先，我们需要导入所需的库：

```py
import warnings
warnings.filterwarnings('ignore')

import random
import math
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
```

# 准备数据集

接下来，我们将生成一些随机数据点，有`500`行和`2`列（*x*和*y*），并将它们用于训练：

```py
data = np.random.randn(500, 2)
```

正如你所看到的，我们的数据有两列：

```py
print data[0]

array([-0.08575873,  0.45157591])
```

第一列表示![](img/66590c20-dfc2-4b48-839f-cc63a9ba1b58.png)值：

```py
print data[0,0]

-0.08575873243708057
```

第二列表示![](img/ab063c3a-cce6-4f37-8037-e30c9a093fff.png)值：

```py
print data[0,1]

0.4515759149158441
```

我们知道简单线性回归方程的表达如下：

![](img/b41dfbb1-d9c9-4016-9c74-1396b9451bbc.png)

因此，我们有两个参数，![](img/bc7de407-91aa-4d28-9b68-8a105971bd42.png)和![](img/1b7b75f5-1fd9-47bd-997f-85319ea76537.png)。我们将这两个参数都存储在名为`theta`的数组中。首先，我们将`theta`初始化为零，如下所示：

```py
theta = np.zeros(2)
```

函数`theta[0]`表示![](img/e9a82001-7e29-40fa-be76-cacaebe1c975.png)的值，而函数`theta[1]`表示![](img/3f8557ef-0294-4299-b920-1fcaf33f9a02.png)的值：

```py
print theta

array([0., 0.])
```

# 定义损失函数

线性回归的**均方误差**（**MSE**）如下所示：

![](img/25a25165-374f-48bb-828f-ccc02eacebf9.png)

这里，![](img/98dad91a-ee90-4832-94da-20e58657f1cd.png) 是训练样本的数量，![](img/214c49e5-d122-4a2a-bc14-36a6d572dbd3.png) 是实际值，![](img/a8e11bef-a35b-46b7-a034-e848dea6ae7f.png) 是预测值。

上述损失函数的实现如下所示。我们将`data`和模型参数`theta`输入损失函数，返回均方误差（MSE）。请记住，`data[,0]`具有一个 ![](img/c54f2a3b-c050-4d6b-aa49-5369f43303c4.png) 值，而`data[,1]`具有一个 ![](img/c3a91968-5b14-4f57-9cb6-a15db386a494.png) 值。类似地，`theta [0]`的值为`m`，而`theta[1]`的值为 ![](img/d0fe8134-2a5f-4033-8df2-e59a6bdb531e.png)。

让我们定义损失函数：

```py
def loss_function(data,theta):
```

现在，我们需要获取 ![](img/4b627c78-968a-4c5b-8894-06f178e5fddd.png) 和 ![](img/5087fbda-762c-449f-9e60-1d5cfb52b799.png) 的值：

```py
    m = theta[0]
    b = theta[1]

    loss = 0
```

我们对每个迭代执行此操作：

```py
    for i in range(0, len(data)):
```

现在，我们得到 ![](img/c723faa4-5fbf-4295-b391-8033c2fd1267.png) 和 ![](img/01de849d-be74-4d24-ae46-84e33f534b8c.png) 的值：

```py
        x = data[i, 0]
        y = data[i, 1]
```

然后，我们预测 ![](img/e112d877-4c86-4eb5-8a57-207d77879881.png) 的值：

```py
        y_hat = (m*x + b)
```

这里，我们按照方程 *(3)* 计算损失：

```py
        loss = loss + ((y - (y_hat)) ** 2)
```

然后，我们计算均方误差：

```py
    mse = loss / float(len(data))

    return mse
```

当我们将随机初始化的`data`和模型参数`theta`输入`loss_function`时，会返回均方损失，如下所示：

```py
loss_function(data, theta)

1.0253548008165727
```

现在，我们需要最小化这个损失。为了最小化损失，我们需要计算损失函数 ![](img/45fcd047-b7d1-48af-9c70-f6f1b39ddc7f.png) 关于模型参数 ![](img/3b49a75c-ac98-4deb-9794-d54352e54182.png) 和 ![](img/7c7dcfb5-e088-4ddc-a962-f3de556ea04e.png) 的梯度，并根据参数更新规则更新参数。首先，我们将计算损失函数的梯度。

# 计算损失函数的梯度

损失函数关于参数 ![](img/92d4859f-5356-498f-af86-ca5a144e87ff.png) 的梯度如下：

![](img/3df6bc2d-615e-4d85-990c-70c6f37a9e7e.png)

损失函数关于参数 ![](img/1baea87c-c5e2-4509-8c22-a0d897bb8a4c.png) 的梯度如下：

![](img/4dbfc095-8767-4469-9b9e-0a2eee952e30.png)

我们定义了一个名为`compute_gradients`的函数，该函数接受输入参数`data`和`theta`，并返回计算得到的梯度：

```py
def compute_gradients(data, theta):
```

现在，我们需要初始化梯度：

```py
    gradients = np.zeros(2)
```

然后，我们需要保存数据点的总数`N`：

```py
    N = float(len(data))  
```

现在，我们可以得到 ![](img/b59cf398-1332-48c2-99fb-1226fb8719b3.png) 和 ![](img/8037f57c-68af-4026-b3ce-55f7b47f76c9.png) 的值：

```py
    m = theta[0]
    b = theta[1]
```

我们对每个迭代执行相同操作：

```py
    for i in range(0, len(data)):
```

然后，我们得到 ![](img/095dfe47-f3a5-491d-9442-dd60338e16a7.png) 和 ![](img/0836a358-33f7-433e-b93b-6cb493da3be0.png) 的值：

```py
        x = data[i, 0]
        y = data[i, 1]
```

现在，我们按照方程 *(4)* 计算损失关于 ![](img/77bc2cf8-0682-4bfb-b6d6-66e98369310e.png) 的梯度：

```py
        gradients[0] += - (2 / N) * x * (y - (( m* x) + b))
```

然后，我们根据方程 *(5)* 计算损失相对于 ![](img/76230e49-4358-4f6b-a92c-a882cf6b66d2.png) 的梯度：

```py
        gradients[1] += - (2 / N) * (y - ((theta[0] * x) + b))
```

我们需要添加`epsilon`以避免零除错误：

```py
    epsilon = 1e-6 
    gradients = np.divide(gradients, N + epsilon)

    return gradients
```

当我们输入我们随机初始化的`data`和`theta`模型参数时，`compute_gradients`函数返回相对于 ![](img/f22c0dc5-90aa-43a5-9066-4d535fe01ef6.png) 即 ![](img/8aac270f-aeba-4a18-b172-5ba9b9ae9873.png)，以及相对于 ![](img/8f3c9fe0-900e-493a-992f-e0962ea44bfc.png) 即 ![](img/bdece2b4-a275-4ac0-9929-27fa8b00ec1a.png) 的梯度，如下所示：

```py
compute_gradients(data,theta)

array([-9.08423989e-05,  1.05174511e-04])
```

# 更新模型参数

现在我们已经计算出梯度，我们需要根据我们的更新规则更新我们的模型参数，如下所示：

![](img/76e81629-8d88-4126-82ac-ba628b371785.png)

![](img/f46ada57-bf52-4003-8dd5-13571e32e5f5.png)

因为我们在`theta[0]`中存储了 ![](img/9414fad6-994b-4f63-beed-5d4a2d3aa118.png)，在`theta[1]`中存储了 ![](img/ee192345-3284-48bc-907c-1203e34d5266.png)，所以我们可以写出我们的更新方程如下：

![](img/cb0718ae-0508-4563-aa40-75297f1ab863.png)

正如我们在前一节中学到的，仅在一个迭代中更新梯度不会导致我们收敛到成本函数的最小值，因此我们需要计算梯度并对模型参数进行多次迭代更新。

首先，我们需要设置迭代次数：

```py
num_iterations = 50000
```

现在，我们需要定义学习率：

```py
lr = 1e-2
```

接下来，我们将定义一个名为`loss`的列表来存储每次迭代的损失：

```py
loss = []
```

在每次迭代中，我们将根据我们的参数更新规则从方程 *(8)* 计算并更新梯度：

```py
theta = np.zeros(2)

for t in range(num_iterations):

    #compute gradients
    gradients = compute_gradients(data, theta)

    #update parameter
    theta = theta - (lr*gradients)

    #store the loss
    loss.append(loss_function(data,theta))
```

现在，我们需要绘制`loss`（`Cost`）函数：

```py
plt.plot(loss)
plt.grid()
plt.xlabel('Training Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent')
```

下图显示了损失（**Cost**）随训练迭代次数减少的情况：

![](img/39abba35-483b-4ff6-b7e7-e898034b0412.png)

因此，我们学会了梯度下降可以用来找到模型的最优参数，然后我们可以用它来最小化损失。在下一节中，我们将学习梯度下降算法的几种变体。

# 梯度下降与随机梯度下降的对比

我们使用我们的参数更新方程 *(1)* 多次更新模型参数，直到找到最优参数值。在梯度下降中，为了执行单个参数更新，我们遍历训练集中的所有数据点。因此，每次更新模型参数时，我们都要遍历训练集中的所有数据点。仅在遍历训练集中的所有数据点后更新模型参数使得梯度下降非常缓慢，并且会增加训练时间，特别是当数据集很大时。

假设我们有一个包含 100 万数据点的训练集。我们知道，我们需要多次更新模型的参数才能找到最优参数值。因此，即使是执行单次参数更新，我们也需要遍历训练集中的所有 100 万数据点，然后更新模型参数。这无疑会使训练变慢。这是因为我们不能仅通过单次更新找到最优参数；我们需要多次更新模型参数才能找到最优值。因此，如果我们对训练集中的每个参数更新都进行 100 万次数据点的迭代，这肯定会减慢我们的训练速度。

因此，为了应对这一情况，我们引入了**随机梯度下降**（**SGD**）。与梯度下降不同，我们无需等待在训练集中迭代所有数据点后更新模型参数；我们只需在遍历训练集中的每一个数据点后更新模型参数。

由于我们在随机梯度下降中在遍历每个单独数据点后更新模型参数，相比梯度下降，它将更快地学习到模型的最优参数，从而缩短训练时间。

SGD 的作用是什么？当我们有一个巨大的数据集时，使用传统的梯度下降方法，我们只在遍历完整个数据集中的所有数据点后更新参数。因此，在整个数据集上进行多次迭代后，我们才能达到收敛，并且显然这需要很长时间。但是，在随机梯度下降中，我们在遍历每个单独的训练样本后更新参数。也就是说，我们从第一个训练样本开始就学习到寻找最优参数，这有助于相比传统的梯度下降方法更快地达到收敛。

我们知道，周期指的是神经网络查看整个训练数据的次数。因此，在梯度下降中，每个周期我们执行参数更新。这意味着，在每个周期结束后，神经网络看到整个训练数据。我们按以下方式每个周期执行参数更新：

![](img/f74c110a-5011-45ad-a74a-258cc6163d99.png)

然而，在随机梯度下降中，我们无需等到每个周期完成后才更新参数。也就是说，我们不需要等到神经网络看到整个训练数据后才更新参数。相反，我们在每个周期中从看到单个训练样本开始就更新网络的参数：

![](img/adc3a4b7-55bb-445a-8144-371d292009e2.png)

下图显示了梯度下降和随机梯度下降如何执行参数更新并找到最小成本。图中心的星号符号表示我们最小成本的位置。正如您所见，随机梯度下降比普通梯度下降更快地达到收敛。您还可以观察到随机梯度下降中梯度步骤的振荡；这是因为我们在每个训练样本上更新参数，因此与普通梯度下降相比，SGD 中的梯度步骤变化频繁：

![](img/9d57ee66-4d0d-4b1d-a9b9-ece39b2d86b0.png)

还有另一种称为**小批量梯度下降**的梯度下降变体。它吸取了普通梯度下降和随机梯度下降的优点。在 SGD 中，我们看到我们为每个训练样本更新模型的参数。然而，在小批量梯度下降中，我们不是在每个训练样本迭代后更新参数，而是在迭代一些数据点的批次后更新参数。假设批量大小为 50，这意味着我们在迭代 50 个数据点后更新模型的参数，而不是在迭代每个单独数据点后更新模型的参数。

下图显示了 SGD 和小批量梯度下降的轮廓图：

![](img/db95290c-235f-4926-a23f-e78df9cc9651.png)

这些梯度下降类型之间的差异如下：

+   **梯度下降**：在训练集中迭代所有数据点后更新模型参数

+   **随机梯度下降**：在训练集中迭代每个单独数据点后更新模型的参数

+   **小批量梯度下降**：在训练集中迭代*n*个数据点后更新模型参数

对于大型数据集，小批量梯度下降优于普通梯度下降和 SGD，因为小批量梯度下降的表现优于其他两种方法。

小批量梯度下降的代码如下所示。

首先，我们需要定义`minibatch`函数：

```py
def minibatch(data, theta, lr = 1e-2, minibatch_ratio = 0.01, num_iterations = 1000):
```

接下来，我们将通过将数据长度乘以`minibatch_ratio`来定义`minibatch_size`：

```py
    minibatch_size = int(math.ceil(len(data) * minibatch_ratio))
```

现在，在每次迭代中，我们执行以下操作：

```py
    for t in range(num_iterations):
```

接下来，选择`sample_size`：

```py
        sample_size = random.sample(range(len(data)), minibatch_size)
        np.random.shuffle(data)
```

现在，基于`sample_size`对数据进行抽样：

```py
        sample_data = data[0:sample_size[0], :]
```

计算`sample_data`相对于`theta`的梯度：

```py
        grad = compute_gradients(sample_data, theta)
```

在计算了给定小批量大小的抽样数据的梯度后，我们按以下方式更新模型参数`theta`：

```py
        theta = theta - (lr * grad)

 return theta
```

# 基于动量的梯度下降

在本节中，我们将学习两种新的梯度下降变体，称为**动量**和 Nesterov 加速梯度。

# 带有动量的梯度下降

我们在 SGD 和小批量梯度下降中遇到了问题，因为参数更新中存在振荡。看看下面的图表，展示了小批量梯度下降是如何达到收敛的。正如您所见，梯度步骤中存在振荡。振荡由虚线表示。您可能注意到，它朝一个方向迈出梯度步骤，然后朝另一个方向，依此类推，直到达到收敛：

![](img/b3701eba-0ceb-4d34-81c4-474e065d9c03.png)

这种振荡是由于我们在迭代每个*n*个数据点后更新参数，更新方向会有一定的变化，从而导致每个梯度步骤中的振荡。由于这种振荡，很难达到收敛，并且减慢了达到收敛的过程。

为了缓解这个问题，我们将介绍一种称为**动量**的新技术。如果我们能够理解梯度步骤达到更快收敛的正确方向，那么我们可以使我们的梯度步骤在那个方向导航，并减少不相关方向上的振荡；也就是说，我们可以减少采取不导致收敛的方向。

那么，我们该如何做呢？基本上，我们从前一梯度步骤的参数更新中获取一部分，并添加到当前梯度步骤中。在物理学中，动量在施加力后使物体保持运动。在这里，动量使我们的梯度保持朝向导致收敛的方向运动。

如果您看一下以下方程，您会看到我们基本上是从上一步的参数更新中取参数更新，![](img/e5b41e74-20a9-4e47-af2c-8986f1a87c88.png)，并将其添加到当前梯度步骤，![](img/1e7bcf4d-985f-4ad8-b1f9-e1ac00461878.png)。我们希望从前一个梯度步骤中获取多少信息取决于因子，即，![](img/51a6304d-576d-4ec3-a26e-664ad0e61caf.png)，以及学习率，用![](img/954de68e-2d2f-4380-9aaa-00deeac3e008.png)表示：

![](img/eb83f676-d492-4491-8ab1-614bdfba32e4.png)

在前述方程中，![](img/b2c9306e-e113-449b-a0ff-e55c2109e875.png) 被称为速度，它加速梯度朝向收敛方向的更新。它还通过在当前步骤中添加来自上一步参数更新的一部分，来减少不相关方向上的振荡。

因此，具有动量的参数更新方程如下表达：

![](img/53ac9a89-4fca-4b09-bbce-267e5ad12953.png)

通过这样做，使用动量进行小批量梯度下降有助于减少梯度步骤中的振荡，并更快地达到收敛。

现在，让我们来看看动量的实现。

首先，我们定义`momentum`函数，如下所示：

```py
def momentum(data, theta, lr = 1e-2, gamma = 0.9, num_iterations = 1000):
```

然后，我们用零初始化`vt`：

```py
    vt = np.zeros(theta.shape[0])
```

下面的代码用于每次迭代覆盖范围：

```py
    for t in range(num_iterations):
```

现在，我们计算相对于`theta`的`gradients`：

```py
        gradients = compute_gradients(data, theta)
```

接下来，我们更新`vt`为![](img/477c810a-9d08-43a5-ab14-d4fde01d0850.png)：

```py
        vt = gamma * vt + lr * gradients
```

现在，我们更新模型参数`theta`，如下所示：

```py
        theta = theta - vt

 return theta
```

# Nesterov 加速梯度

动量法的一个问题是可能会错过最小值。也就是说，当我们接近收敛（最小点）时，动量的值会很高。当动量值在接近收敛时很高时，实际上动量会推动梯度步长变高，并且可能错过实际的最小值；也就是说，当动量在接近收敛时很高时，它可能会超出最小值，如下图所示：

![](img/65dae543-ef11-42d3-9c8c-5bcea24c3d81.png)

为了克服这一问题，Nesterov 引入了一种新的方法，称为**Nesterov 加速梯度**（**NAG**）。

Nesterov 动量背后的基本动机是，我们不是在当前位置计算梯度，而是在动量将带我们到的位置计算梯度，我们称之为前瞻位置。

那么，这到底意味着什么呢？在*带动量的梯度下降*部分，我们了解到以下方程：

![](img/059fe057-0350-4fb5-85fe-51f27aee3500.png)

上述方程告诉我们，我们基本上是用前一步参数更新的一部分来推动当前的梯度步长`vt`到一个新位置，这将帮助我们达到收敛。然而，当动量很高时，这个新位置实际上会超过最小值。

因此，在使用动量进行梯度步进并到达新位置之前，如果我们理解动量将带我们到哪个位置，我们就可以避免超出最小值。如果我们发现动量会带我们到实际错过最小值的位置，那么我们可以减缓动量并尝试达到最小值。

但是我们如何找到动量将带我们到的位置呢？在方程*(2)*中，我们不是根据当前梯度步长`vt`计算梯度，而是根据`vt`的前瞻位置计算梯度。术语`1`基本上告诉我们我们下一个梯度步长的大致位置，我们称之为前瞻位置。这给了我们一个关于下一个梯度步长将在哪里的想法。

因此，我们可以根据 NAG 重新编写我们的方程`vt`如下：

![](img/2801d3ce-6ad0-4ac9-894f-db0adbdd79d7.png)

我们更新我们的参数如下：

![](img/b08cded8-6f68-4913-b47e-ac42ad3b9c74.png)

使用上述方程更新参数可以通过在梯度步骤接近收敛时减慢动量来防止错过最小值。Nesterov 加速方法的实现如下。

首先，我们定义`NAG`函数：

```py
def NAG(data, theta, lr = 1e-2, gamma = 0.9, num_iterations = 1000):
```

然后，我们用零初始化`vt`的值：

```py
    vt = np.zeros(theta.shape[0])
```

对于每次迭代，我们执行以下步骤：

```py
    for t in range(num_iterations):
```

现在，我们需要计算相对于![](img/c47a48e3-6f6b-4ff0-8ae5-b215be5228bc.png)的梯度：

```py
        gradients = compute_gradients(data, theta - gamma * vt)
```

然后，我们将`vt`更新为![](img/8ce44c3a-e42b-4a47-b1ab-728842010ff6.png)：

```py
        vt = gamma * vt + lr * gradients
```

现在，我们将模型参数`theta`更新为![](img/6f0cfa20-7bf8-4250-8dd8-d9bd7ae53190.png)：

```py
        theta = theta - vt

    return theta
```

# 梯度下降的自适应方法

在本节中，我们将学习几种梯度下降的自适应版本。

# 使用 Adagrad 自适应设置学习率

当我们构建深度神经网络时，会有很多参数。参数基本上是网络的权重，所以当我们构建具有多层的网络时，会有很多权重，比如![](img/921bdcf6-8faf-4582-98bd-5aa0143a103e.png)。我们的目标是找到所有这些权重的最优值。在我们之前学到的所有方法中，网络参数的学习率是一个常见值。然而，**Adagrad**（自适应梯度的简称）会根据参数自适应地设置学习率。

经常更新或梯度较大的参数将具有较慢的学习率，而更新不频繁或梯度较小的参数也将具有较慢的学习率。但是我们为什么要这样做？这是因为更新不频繁的参数意味着它们没有被充分训练，所以我们为它们设置较高的学习率；而频繁更新的参数意味着它们已经足够训练，所以我们为它们设置较低的学习率，以防止超出最小值。

现在，让我们看看 Adagrad 如何自适应地改变学习率。之前，我们用![](img/0eb9264a-0aff-4e98-9416-98201ca0f47b.png)表示梯度。为简单起见，在本章中，我们将用![](img/8e969c03-6112-4f27-bef1-7152b8e194c9.png)代表梯度。因此，参数![](img/a03c809e-24b3-4c69-998b-b13ef1878d0d.png)在迭代![](img/8343d84a-b952-4bbf-b4fb-e81e0f63bcad.png)时的梯度可以表示为：

![](img/e696fced-9b54-41eb-9cc6-ae64180154bb.png)

因此，我们可以用![](img/dfde1993-0b27-4d94-908e-44d8452e8bc6.png)来重新编写我们的更新方程，作为梯度符号表示如下：

![](img/8a17fafe-c7e0-4222-a3db-e6a25d8483c8.png)

现在，对于每次迭代![](img/b848153f-c88b-42de-abb7-fa2f30ed7c90.png)，要更新参数![](img/9954b549-05fa-45fa-b92c-c9cbcd3d9dca.png)，我们将学习率除以参数![](img/e2bdd119-534b-4584-8954-5cf72f9c4495.png)的所有先前梯度的平方和，如下所示：

![](img/143ea471-82eb-4eaf-82e3-a9a5e35b60ee.png)

在这里，![](img/a030af86-9b17-4871-972a-a78ca31c12b6.png) 表示参数![](img/545b96da-57b3-4919-9e41-50617a0db779.png)所有先前梯度的平方和。我们添加![](img/2bb4c014-84ed-47e4-9403-7740622fd273.png) 只是为了避免除以零的错误。通常将![](img/82a50656-7a02-443d-bd1c-03998473e2b3.png) 的值设置为一个小数。这里出现的问题是，为什么我们要把学习率除以所有先前梯度的平方和？

我们了解到，具有频繁更新或高梯度的参数将具有较慢的学习率，而具有不频繁更新或小梯度的参数也将具有较高的学习率。

总和，![](img/daa6a6d7-af78-45ce-988d-e3e226db103a.png)，实际上缩放了我们的学习率。也就是说，当过去梯度的平方和值较高时，我们基本上将学习率除以一个高值，因此我们的学习率将变得较低。类似地，如果过去梯度的平方和值较低，则我们将学习率除以一个较低的值，因此我们的学习率值将变高。这意味着学习率与参数的所有先前梯度的平方和成反比。

在这里，我们的更新方程表示如下：

![](img/ca780a96-e7a8-4e50-9f0f-1abcc686d073.png)

简而言之，在 Adagrad 中，当先前梯度值高时，我们将学习率设置为较低的值，当过去梯度值较低时，我们将学习率设置为较高的值。这意味着我们的学习率值根据参数的过去梯度更新而变化。

现在我们已经学习了 Adagrad 算法的工作原理，让我们通过实现它来加深我们的知识。Adagrad 算法的代码如下所示。

首先，定义`AdaGrad`函数：

```py
def AdaGrad(data, theta, lr = 1e-2, epsilon = 1e-8, num_iterations = 10000):
```

定义名为`gradients_sum`的变量来保存梯度和，并将它们初始化为零：

```py
    gradients_sum = np.zeros(theta.shape[0])
```

对于每次迭代，我们执行以下步骤：

```py
    for t in range(num_iterations):
```

然后，我们计算损失对`theta`的梯度：

```py
        gradients = compute_gradients(data, theta) 
```

现在，我们计算梯度平方和，即![](img/8f690939-5865-4088-9bf5-13525f26c9af.png)：

```py
        gradients_sum += gradients ** 2
```

之后，我们计算梯度更新，即![](img/b934756d-4983-4ce3-995e-5886ca0adc58.png)：

```py
        gradient_update = gradients / (np.sqrt(gradients_sum + epsilon))
```

现在，更新`theta`模型参数，使其为![](img/a76a05a7-2b37-4d98-86f5-6305bc9a7c31.png) ：

```py
        theta = theta - (lr * gradient_update)

    return theta
```

然而，Adagrad 方法存在一个缺点。在每次迭代中，我们都会累积和求和所有过去的平方梯度。因此，在每次迭代中，过去平方梯度值的总和将会增加。当过去平方梯度值的总和较高时，分母中会有一个较大的数。当我们将学习率除以一个非常大的数时，学习率将变得非常小。因此，经过几次迭代后，学习率开始衰减并变成一个无限小的数值——也就是说，我们的学习率将单调递减。当学习率降至一个非常低的值时，收敛需要很长时间。

在下一节中，我们将看到 Adadelta 如何解决这个缺点。

# 采用 Adadelta 方法摒弃学习率

Adadelta 是 Adagrad 算法的增强版。在 Adagrad 中，我们注意到学习率减小到一个非常低的数字的问题。虽然 Adagrad 能够自适应地学习学习率，但我们仍然需要手动设置初始学习率。然而，在 Adadelta 中，我们根本不需要学习率。那么，Adadelta 算法是如何学习的呢？

在 Adadelta 中，我们不是取所有过去的平方梯度的总和，而是设定一个大小为 ![](img/d5c1b640-6064-4614-94c3-4711517041e8.png) 的窗口，并仅从该窗口中取过去的平方梯度的总和。在 Adagrad 中，我们取所有过去的平方梯度的总和，并导致学习率减小到一个低数字。为了避免这种情况，我们只从一个窗口内取过去的平方梯度的总和。

如果 ![](img/7501c19e-7f5b-4007-9c3d-8bdd403bf972.png) 是窗口大小，则我们的参数更新方程如下所示：

![](img/7d472342-c741-47d0-a9f3-c5df6d65ba19.png)

然而，问题在于，虽然我们仅从一个窗口内取梯度，![](img/2ea21248-7d34-45bb-9ad0-c97848dfee51.png)，但在每次迭代中将窗口内所有梯度进行平方并存储是低效的。因此，我们可以采取梯度的运行平均值，而不是这样做。

我们通过将先前的梯度的运行平均值 ![](img/bbb28974-77d5-48b2-92bb-f82995761522.png) 和当前梯度 ![](img/cc6925a0-661f-4248-90ba-21adadf5485b.png) 相加来计算迭代 *t* 的梯度的运行平均值 ![](img/5a963575-25d4-4ca2-b5fe-065ad4351973.png)：

![](img/12f4e9d1-20aa-49d7-903a-a0315ba86207.png)

不仅仅是取运行平均值，我们还采取梯度的指数衰减运行平均值，如下所示：

![](img/c1b0f509-4621-4956-a49d-15a19723991a.png)

在这里，![](img/77fc07fe-f0d6-4f74-a7f3-0270b2ff4fcf.png) 被称为指数衰减率，类似于我们在动量中看到的那个——用于决定从前一次梯度的运行平均值中添加多少信息。

现在，我们的更新方程如下所示：

![](img/d800a845-64f6-4472-bc96-7b8153deccaf.png)

为了简化符号，让我们将![](img/7e80684e-53c8-4895-a6ea-c820e2bd63c1.png)记作![](img/8ca829a5-4620-440f-996e-b7b673987d5c.png)，这样我们可以将前一个更新方程重写为如下形式：

![](img/aaa3abba-608c-4f8d-9597-72c0697b0e6e.png)

根据前述方程，我们可以推断如下：

![](img/383befa7-159b-41d5-aa26-194b0765a082.png)

如果你观察前一个方程中的分母，我们基本上在计算迭代过程中梯度的均方根，![](img/c2272d39-a5b1-438a-8358-fa14ea1c5c1f.png)，因此我们可以简单地用以下方式写出：

![](img/4725e93c-7d92-4c8b-8064-632079ceb665.png)

通过将方程 *(13)* 代入方程 *(12)*，我们可以写出如下：

![](img/4f5ec229-a979-4ab0-9ebf-b780c0189982.png)

然而，在我们的方程中仍然有学习率，![](img/7db8a465-4cf4-48dc-a7f4-e4273fee1c6b.png)，术语。我们如何摆脱它？我们可以通过使参数更新的单位与参数相符来实现。正如你可能注意到的那样，![](img/9c736c49-b827-4714-ae7e-662a26876719.png)和![](img/43cfa07c-dafd-46b8-bd4b-7003ded49a8e.png)的单位并不完全匹配。为了解决这个问题，我们计算参数更新的指数衰减平均值，![](img/dc80e9c3-6926-4ca2-92fc-a7eaacbd810e.png)，正如我们在方程 *(10)* 中计算了梯度的指数衰减平均值，![](img/6efc0ba3-2eb1-4a95-ba1f-75088f554f52.png)。因此，我们可以写出如下：

![](img/d1b38486-b79e-4a08-8954-b46cecc10789.png)

它类似于梯度的 RMS，![](img/74127744-93af-4dfc-bec6-57299b903588.png)，类似于方程 *(13)*。我们可以将参数更新的 RMS 写成如下形式：

![](img/ce6f12d2-613a-4583-be31-93c4005b9be2.png)

然而，参数更新的 RMS 值，![](img/1f7ce6e6-cada-49fe-9014-08621044a329.png)，是未知的，即![](img/5bdfb640-f031-4aad-bff5-860c7b834fbd.png)是未知的，因此我们可以通过考虑直到上一个更新，![](img/947ae65b-23c0-423b-a8c1-35c3ebe5a374.png)，来近似计算它。

现在，我们只需用参数更新的 RMS 值取代学习率。也就是说，我们在方程 *(14)* 中用![](img/f5044369-2aeb-4887-9afd-0098598987d9.png)取代![](img/076e95dd-a926-4ef4-911b-cf48ceb58891.png)，并写出如下：

![](img/2751d98e-b867-4cfa-9244-8c3a0821fbe9.png)

将方程 *(15)* 代入方程 *(11)*，我们的最终更新方程如下：

![](img/17dd5b3c-fba9-4b5d-b575-fd73ac204e68.png)

![](img/9bd169af-8202-43f9-991a-167b7682fa27.png)

现在，让我们通过实现了解 Adadelta 算法。

首先，我们定义`AdaDelta`函数：

```py
def AdaDelta(data, theta, gamma = 0.9, epsilon = 1e-5, num_iterations = 1000):
```

然后，我们将`E_grad2`变量初始化为零，用于存储梯度的运行平均值，并将`E_delta_theta2`初始化为零，用于存储参数更新的运行平均值，如下所示：

```py
    # running average of gradients
    E_grad2 = np.zeros(theta.shape[0])

    #running average of parameter update
    E_delta_theta2 = np.zeros(theta.shape[0])
```

每次迭代，我们执行以下步骤：

```py
    for t in range(num_iterations):
```

现在，我们需要计算相对于`theta`的`gradients`：

```py
        gradients = compute_gradients(data, theta) 
```

然后，我们可以计算梯度的运行平均值：

```py
        E_grad2 = (gamma * E_grad2) + ((1\. - gamma) * (gradients ** 2))
```

在这里，我们将计算`delta_theta`，即，![](img/6082dff2-d77b-4a67-8d43-094b9c2dc17f.png)：

```py
        delta_theta = - (np.sqrt(E_delta_theta2 + epsilon)) / (np.sqrt(E_grad2 + epsilon)) * gradients
```

现在，我们可以计算参数更新的运行平均值，![](img/fd64714f-4f7f-477d-a2e2-e9796e2ea107.png)：

```py
        E_delta_theta2 = (gamma * E_delta_theta2) + ((1\. - gamma) * (delta_theta ** 2))
```

接下来，我们将更新模型参数`theta`，使其变为![](img/e0a32cbd-f87c-4f52-981d-7fa487762814.png)：

```py
        theta = theta + delta_theta

    return theta
```

# 通过 RMSProp 克服 Adagrad 的限制

与 Adadelta 类似，RMSProp 被引入来解决 Adagrad 的学习率衰减问题。因此，在 RMSProp 中，我们如下计算梯度的指数衰减运行平均值：

![](img/70a39013-89e8-4d74-b5d2-af6198ba6446.png)

而不是取所有过去梯度的平方和，我们使用这些梯度的运行平均值。这意味着我们的更新方程如下：

![](img/95aacb08-98ff-4112-af96-544e87f5422e.png)

建议将学习率![](img/0b5604c4-d979-49d8-ab84-f266667632fb.png)设置为`0.9`。现在，我们将学习如何在 Python 中实现 RMSProp。

首先，我们需要定义`RMSProp`函数：

```py
def RMSProp(data, theta, lr = 1e-2, gamma = 0.9, epsilon = 1e-6, num_iterations = 1000):
```

现在，我们需要用零来初始化`E_grad2`变量，以存储梯度的运行平均值：

```py
    E_grad2 = np.zeros(theta.shape[0])
```

每次迭代时，我们执行以下步骤：

```py
    for t in range(num_iterations):
```

然后，我们计算相对于`theta`的`gradients`：

```py
        gradients = compute_gradients(data, theta) 
```

接下来，我们计算梯度的运行平均值，即，![](img/8b2a51a3-3e9f-4e6b-9390-7dc65e8dc37a.png)：

```py
        E_grad2 = (gamma * E_grad2) + ((1\. - gamma) * (gradients ** 2))
```

现在，我们更新模型参数`theta`，使其变为![](img/9dbe5cdf-01b4-4668-afcf-6cbb72003eba.png)：

```py
        theta = theta - (lr / (np.sqrt(E_grad2 + epsilon)) * gradients)
    return theta
```

# 自适应矩估计

**自适应矩估计**，简称**Adam**，是优化神经网络中最常用的算法之一。在阅读有关 RMSProp 的内容时，我们了解到，为了避免学习率衰减问题，我们计算了平方梯度的运行平均值：

![](img/b3f51f1d-6ade-492e-aefc-61e6c6ed5c25.png)

RMSprop 的最终更新方程如下所示：

![](img/d1ea583f-7ebd-46e7-97c9-c7e426a35c2e.png)

类似于此，在 Adam 中，我们还计算平方梯度的运行平均值。然而，除了计算平方梯度的运行平均值外，我们还计算梯度的运行平均值。

梯度的运行平均值如下所示：

![](img/34ab5642-f504-4752-961d-143f1c859e11.png)

平方梯度的运行平均值如下所示：

![](img/d2a20842-c858-4d22-95d1-998e5419c757.png)

由于许多文献和库将 Adam 中的衰减率表示为 ![](img/df2124c0-d034-4ed9-96db-054189b46924.png)，而不是 ![](img/363da8bd-bf34-46d1-b780-f2e525aacceb.png)，我们也将使用 ![](img/24d59526-7084-423f-a6a1-08fea1dccecc.png) 来表示 Adam 中的衰减率。因此，在方程式 *(16)* 和 *(17)* 中，![](img/c6812bc9-7617-4c73-8196-42cbf1358429.png) 和 ![](img/eba646cb-bdf2-4046-94a7-179bf57eda6f.png) 分别表示梯度运行平均值和平方梯度的指数衰减率。

因此，我们的更新方程式变为以下形式：

![](img/a9e421e2-9551-4936-9545-325f7c2063d9.png)

梯度运行平均值和平方梯度的运行平均值基本上是这些梯度的第一和第二时刻。也就是说，它们分别是我们梯度的均值和未中心化方差。为了符号简洁起见，让我们将 ![](img/43b1f7ac-39b2-40e2-8878-a79457c93ed5.png) 表示为 ![](img/05c11c4a-cfa8-481c-a11e-2a93745c3ab9.png)，将 ![](img/251767bb-33e2-4ed5-87b6-b50ac865576d.png) 表示为 ![](img/bfa9ed56-26be-4646-92d5-db97d0355fe6.png)。

因此，我们可以将方程 *(16)* 和 *(17)* 重写如下：

![](img/b55ff174-f556-4654-b015-5ceb1446b637.png)

![](img/13ed75a5-2581-47ee-93e8-aa092bd94151.png)

我们首先将初始时刻估计设置为零。也就是说，我们用零初始化 ![](img/010c97de-58b7-4b89-83fa-792dd5765de4.png) 和 ![](img/a3c4b67f-174d-4436-958e-a7cc63f150c4.png)。当初始估计设置为 0 时，即使经过多次迭代后它们仍然非常小。这意味着它们会偏向 0，特别是当 ![](img/e97e4a53-38c4-4acd-b55a-73192dbe7cd5.png) 和 ![](img/238ebaf3-b706-4feb-8165-4e4385a92e20.png) 接近 1 时。因此，为了抵消这种影响，我们通过仅将它们除以 ![](img/3afa1ee2-0ab9-41cf-b67b-57c433323f37.png) 来计算 ![](img/5b150a4c-513b-4491-8122-771f492ba611.png) 和 ![](img/2be81116-58a9-4bd2-9006-7893608d63f6.png) 的偏差校正估计，如下所示：

![](img/1aa781fc-0e50-4598-b403-635dbb989ab4.png)

![](img/7eae202a-5884-43c6-be3f-a99dad9c0401.png)

在这里，![](img/b5328637-a5cd-4996-afa9-b53378a30465.png) 和 ![](img/fcade50e-ce09-46db-8c7e-e2f5080ff36a.png) 分别是 ![](img/8b11788d-7089-4728-8c3c-b8f31f713db4.png) 和 ![](img/98cb351b-581c-4cee-8c75-873a17945df2.png) 的偏差校正估计。

因此，我们的最终更新方程式如下：

![](img/6246a7f0-a18e-4932-90f1-fc6bc3558b6a.png)

现在，让我们了解如何在 Python 中实现 Adam。

首先，让我们定义 `Adam` 函数如下：

```py
def Adam(data, theta, lr = 1e-2, beta1 = 0.9, beta2 = 0.9, epsilon = 1e-6, num_iterations = 1000):
```

然后，我们用 `zeros` 初始化第一时刻 `mt` 和第二时刻 `vt`：

```py
    mt = np.zeros(theta.shape[0])
    vt = np.zeros(theta.shape[0])
```

每次迭代，我们执行以下步骤：

```py
    for t in range(num_iterations):
```

接下来，我们计算相对于 `theta` 的 `gradients`：

```py
        gradients = compute_gradients(data, theta) 
```

然后，我们更新第一时刻 `mt`，使其为 ![](img/6479d95b-8005-4966-984d-3200dab96bed.png)：

```py
        mt = beta1 * mt + (1\. - beta1) * gradients
```

接下来，我们更新第二矩`vt`，使其为![](img/dcdf171a-595f-4074-8ec9-25495fc6f7af.png)：

```py
        vt = beta2 * vt + (1\. - beta2) * gradients ** 2
```

现在，我们计算偏差校正估计的`mt`，即![](img/9fed01d0-11bd-44c4-88d8-27ebc5ae8b3a.png)：

```py
        mt_hat = mt / (1\. - beta1 ** (t+1))
```

接下来，我们计算偏差校正估计的`vt`，即![](img/5452eea8-5aa0-493d-ad9b-dc39f12b950d.png)：

```py
        vt_hat = vt / (1\. - beta2 ** (t+1))
```

最后，我们更新模型参数`theta`，使其为![](img/182717ec-ad2c-4820-8c06-bca033b45355.png)：

```py
        theta = theta - (lr / (np.sqrt(vt_hat) + epsilon)) * mt_hat

    return theta
```

# Adamax – 基于无穷范数的 Adam

现在，我们将看一个叫**Adamax**的 Adam 算法的一个小变体。让我们回忆一下 Adam 中的二阶矩方程：

![](img/09810da8-e2a4-4c52-80b6-218dcf0b869f.png)

正如您从上述方程看到的那样，我们将梯度按当前和过去梯度的![](img/396097e1-c5af-4252-9fcf-173eb004ebc0.png)范数的倒数进行缩放（![](img/96e6a50e-e40a-4999-a9cb-73c4e08807cc.png)范数基本上是值的平方）：

![](img/d7361a21-98a0-45a1-9359-0241675a7e6b.png)

而不仅仅是有![](img/9272cabb-f165-45d8-b836-a0f844855bb2.png)，我们能将它推广到![](img/b4a2e9d9-e01b-4d9e-a6ad-db8ab7a35230.png)范数吗？一般情况下，当我们有大![](img/9bec027b-eb90-4c01-abcd-ed54ea819ef3.png)范数时，我们的更新会变得不稳定。然而，当我们设置![](img/f0c92e11-97fb-4943-bfc2-631b36c7afc7.png)值为![](img/c8cc7775-1fde-4ee3-993b-dcfebe9e3e6e.png)，即![](img/33a528ff-3c8c-4150-a60b-c4ff0dc4e906.png)时，![](img/c4f4e684-6be1-4562-a631-f666c519f1dd.png)方程变得简单且稳定。我们不仅仅是对梯度参数化，![](img/004e5fdb-7b24-42e9-b848-3be63db54ee7.png)，我们还对衰减率，![](img/daf1f97c-fa96-4224-965a-36f4aea4133b.png)，进行参数化。因此，我们可以写出以下内容：

![](img/f6a4894a-9edd-4f41-b99a-1c73cef1beb3.png)

当我们设置限制时，![](img/afef1e2b-a4ff-4a68-8b56-c5ec148182a2.png)趋向于无穷大，然后我们得到以下最终方程：

![](img/cb48ebc1-f94e-47ee-91c0-dfaf45ac8dbb.png)

您可以查阅本章末尾列出的*进一步阅读*部分的论文，了解这是如何推导出来的。

我们可以将上述方程重写为一个简单的递归方程，如下所示：

![](img/0d41169c-0374-4b9f-9805-cf672f7ce1fe.png)

计算![](img/b3f21b89-9eaa-4591-838a-fdc425011972.png)类似于我们在*自适应动量估计*部分看到的，因此我们可以直接写出以下内容：

![](img/d31edaf5-271b-488a-9cb6-00a78ee21f87.png)

通过这样做，我们可以计算偏差校正估计的![](img/dfe1e273-e827-4d73-9a5c-e77be289e10d.png)：

![](img/3b1a70b5-d3e7-4dcc-95cb-85d1caa57a86.png)

因此，最终的更新方程变为以下形式：

![](img/86e9d4cf-a048-4356-9a9a-3ddc03595084.png)

为了更好地理解 Adamax 算法，让我们逐步编写代码。

首先，我们定义`Adamax`函数，如下所示：

```py
def Adamax(data, theta, lr = 1e-2, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-6, num_iterations = 1000):
```

然后，我们用零来初始化第一时刻`mt`和第二时刻`vt`：

```py
    mt = np.zeros(theta.shape[0])
    vt = np.zeros(theta.shape[0])
```

对于每次迭代，我们执行以下步骤：

```py
    for t in range(num_iterations):
```

现在，我们可以计算关于`theta`的梯度，如下所示：

```py
        gradients = compute_gradients(data, theta) 
```

然后，我们计算第一时刻`mt`，如下所示：

```py
        mt = beta1 * mt + (1\. - beta1) * gradients
```

接下来，我们计算第二时刻`vt`，如下所示：

```py
        vt = np.maximum(beta2 * vt, np.abs(gradients))
```

现在，我们可以计算`mt`的偏差校正估计，即![](img/b27f19b9-3357-4b19-a9cf-367b59e880b3.png)：

```py
        mt_hat = mt / (1\. - beta1 ** (t+1))
```

更新模型参数`theta`，使其为![](img/e61b483b-34ae-4351-940e-eda822970d60.png)：

```py
        theta = theta - ((lr / (vt + epsilon)) * mt_hat)

    return theta
```

# 使用 AMSGrad 的自适应矩估计

Adam 算法的一个问题是有时无法达到最优收敛，或者它达到次优解。已经注意到，在某些情况下，Adam 无法实现收敛，或者达到次优解，而不是全局最优解。这是由于指数移动平均梯度。记得我们在 Adam 中使用梯度的指数移动平均来避免学习率衰减的问题吗？

然而，问题在于由于我们采用梯度的指数移动平均，我们错过了不经常出现的梯度信息。

为了解决这个问题，AMSGrad 的作者对 Adam 算法进行了微小的修改。回想一下我们在 Adam 中看到的二阶矩估计，如下所示：

![](img/50fd89e6-3336-416a-8acd-4b361c5dd11d.png)

在 AMSGrad 中，我们使用稍微修改过的![](img/c60f3dab-10e9-4e71-88ef-16e13e8b209e.png)的版本。我们不直接使用![](img/b3dc98f8-8596-48c3-a11c-16c8d704963b.png)，而是取直到前一步的![](img/b64abff4-419a-48b5-993f-305d0f29bc14.png)的最大值，如下所示：

![](img/1e77b55f-f6b1-4821-92d6-3ba7af3de434.png)

保留了由于指数移动平均而保留信息梯度，而不是逐步淘汰。

因此，我们的最终更新方程式如下所示：

![](img/924a11d1-0148-42c3-90d2-759d475eb0e8.png)

现在，让我们了解如何在 Python 中编写 AMSGrad。

首先，我们定义`AMSGrad`函数，如下所示：

```py
def AMSGrad(data, theta, lr = 1e-2, beta1 = 0.9, beta2 = 0.9, epsilon = 1e-6, num_iterations = 1000):
```

然后，我们初始化第一时刻`mt`、第二时刻`vt`以及修改后的`vt`，即`vt_hat`，均为`zeros`，如下所示：

```py
    mt = np.zeros(theta.shape[0])
    vt = np.zeros(theta.shape[0])
    vt_hat = np.zeros(theta.shape[0])
```

对于每次迭代，我们执行以下步骤：

```py
    for t in range(num_iterations):
```

现在，我们可以计算关于`theta`的梯度：

```py
        gradients = compute_gradients(data, theta) 
```

然后，我们计算第一时刻`mt`，如下所示：

```py
        mt = beta1 * mt + (1\. - beta1) * gradients
```

接下来，我们更新第二时刻`vt`，如下所示：

```py
       vt = beta2 * vt + (1\. - beta2) * gradients ** 2
```

在 AMSGrad 中，我们使用稍微修改过的![](img/b905b063-1b90-4268-a01b-685c63321a23.png)的版本。我们不直接使用![](img/cf6ace96-be36-43a2-9fb2-e8617e10e286.png)，而是取直到前一步的![](img/db094c80-9b62-49d5-a88a-169a38891439.png)的最大值。因此，![](img/103a5249-9849-463d-b5ab-a3dad1bb5ad9.png)的实现如下：

```py
        vt_hat = np.maximum(vt_hat,vt)
```

在这里，我们将计算`mt`的偏差校正估计，即![](img/1f438fc5-4a60-4d4f-ac85-e24b8754e614.png)：

```py
        mt_hat = mt / (1\. - beta1 ** (t+1))
```

现在，我们可以更新模型参数`theta`，使其为![](img/882d17d3-e9e7-4100-9dbb-294a078c3e94.png)：

```py
          theta = theta - (lr / (np.sqrt(vt_hat) + epsilon)) * mt_hat

    return theta
```

# Nadam – 将 NAG 添加到 ADAM 中

Nadam 是 Adam 方法的另一个小扩展。正如其名称所示，在这里，我们将 NAG 合并到 Adam 中。首先，让我们回顾一下我们在 Adam 中学到的内容。

我们按以下方式计算第一和第二时刻：

![](img/a1e7996d-a2e6-4053-b993-2d597c4dae22.png)

![](img/e478ac82-f544-4b2f-9723-02d6999951c4.png)

然后，我们计算第一和第二时刻的偏差校正估计，如下：

![](img/e1e3821e-6433-432d-9150-6cbf08fc361b.png)

![](img/cd9f08a6-3fca-4e28-84d8-d718285b8670.png)

我们 Adam 的最终更新方程式表达如下：

![](img/d3bed132-5f38-4a36-8c7d-94ba42e4f4de.png)

现在，我们将看看 Nadam 如何修改 Adam 以使用 Nesterov 动量。在 Adam 中，我们计算第一时刻如下：

![](img/80c74cbe-8d42-4148-a948-92f11e10163b.png)

我们将这个第一时刻更改为 Nesterov 加速动量。也就是说，我们不再使用先前的动量，而是使用当前的动量，并将其用作前瞻：

![](img/76be277c-6543-4968-9bd4-aa8d1044a6a3.png)

我们无法像在 Adam 中计算偏差校正估计那样在这里计算，因为这里的![](img/d6171a5c-f432-4255-b665-a24aaa0d2540.png)来自当前步骤，而![](img/3431aced-2e2a-4450-8d1e-c280f09ba27f.png)来自后续步骤。因此，我们改变偏差校正估计步骤如下：

![](img/2deb0295-245d-4ab1-9fe1-2c613d4a16b8.png)

![](img/b7d13f32-8f97-4090-9910-4c759de7458a.png)

因此，我们可以将我们的第一时刻方程重写为以下形式：

![](img/da91ed55-f650-49c7-bbd6-9bf7b70d3d9e.png)

因此，我们的最终更新方程变为以下形式：

![](img/182c9f72-90ce-4b54-9413-164cbda3e08d.png)

现在让我们看看如何在 Python 中实现 Nadam 算法。

首先，我们定义`nadam`函数：

```py
def nadam(data, theta, lr = 1e-2, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-6, num_iterations = 500):
```

然后，我们用零初始化第一时刻`mt`和第二时刻`vt`：

```py
    mt = np.zeros(theta.shape[0])
    vt = np.zeros(theta.shape[0])
```

接下来，我们将`beta_prod`设置为`1`：

```py
    beta_prod = 1
```

对于每次迭代，我们执行以下步骤：

```py
    for t in range(num_iterations):
```

然后，我们计算相对于`theta`的梯度：

```py

        gradients = compute_gradients(data, theta)
```

然后，我们计算第一时刻`mt`，使其为![](img/a90248e7-e1f2-4d29-98c6-1faff1bdd3f9.png)：

```py
        mt = beta1 * mt + (1\. - beta1) * gradients
```

现在，我们可以更新第二时刻`vt`，使其为![](img/656e0070-28a6-413d-aea9-44b30bd53e98.png)：

```py
       vt = beta2 * vt + (1\. - beta2) * gradients ** 2
```

现在，我们计算`beta_prod`，即![](img/25815526-3f96-4e89-91a9-0c89924cfbb6.png)：

```py
        beta_prod = beta_prod * (beta1)
```

接下来，我们计算`mt`的偏差校正估计，使其为![](img/dbc9d845-d338-4041-af28-17b321535aaf.png)：

```py
        mt_hat = mt / (1\. - beta_prod)
```

然后，我们计算`gt`的偏差校正估计，使其为![](img/90ea0452-143c-417c-8d03-65ee10e2ae87.png)：

```py
        g_hat = grad / (1\. - beta_prod)
```

从这里开始，我们计算`vt`的偏差校正估计，使其为![](img/0fef7a9a-f152-4f65-bea8-a9166841b801.png)：

```py
        vt_hat = vt / (1\. - beta2 ** (t))
```

现在，我们计算`mt_tilde`，使其为![](img/cb21eefa-02cd-4aaf-a638-d45f05ca767d.png)：

```py
        mt_tilde = (1-beta1**t+1) * mt_hat + ((beta1**t)* g_hat)
```

最后，我们通过使用![](img/7026ecc8-19aa-4955-a1a3-9285b9b341df.png)来更新模型参数`theta`：

```py
        theta = theta - (lr / (np.sqrt(vt_hat) + epsilon)) * mt_hat

 return theta
```

通过这样做，我们学习了用于训练神经网络的各种流行的梯度下降算法变体。执行包含所有回归变体的完整代码的 Jupyter Notebook 可以在[`bit.ly/2XoW0vH`](http://bit.ly/2XoW0vH)找到。

# 摘要

我们从学习什么是凸函数和非凸函数开始本章。然后，我们探讨了如何使用梯度下降找到函数的最小值。我们学习了梯度下降通过计算最优参数来最小化损失函数。后来，我们看了 SGD，其中我们在迭代每个数据点之后更新模型的参数，然后我们学习了小批量 SGD，其中我们在迭代一批数据点之后更新参数。

继续前进，我们学习了如何使用动量来减少梯度步骤中的振荡并更快地达到收敛。在此之后，我们了解了 Nesterov 动量，其中我们不是在当前位置计算梯度，而是在动量将我们带到的位置计算梯度。

我们还学习了 Adagrad 方法，其中我们为频繁更新的参数设置了低学习率，对不经常更新的参数设置了高学习率。接下来，我们了解了 Adadelta 方法，其中我们完全放弃了学习率，而是使用梯度的指数衰减平均值。然后，我们学习了 Adam 方法，其中我们使用第一和第二动量估计来更新梯度。

在此之后，我们探讨了 Adam 的变体，如 Adamax，我们将 Adam 的![](img/a4a9a09c-4873-4dd0-a000-91b81420b218.png)范数泛化为![](img/d99a407c-5386-45d9-b34b-f61814aa6b12.png)，以及 AMSGrad，我们解决了 Adam 达到次优解的问题。在本章的最后，我们学习了 Nadam，其中我们将 Nesterov 动量整合到 Adam 算法中。

在下一章中，我们将学习一种最广泛使用的深度学习算法之一，称为**循环神经网络**（**RNNs**），以及如何使用它们来生成歌词。

# 问题

通过回答以下问题来回顾梯度下降：

1.  SGD 与普通梯度下降有什么区别？

1.  解释小批量梯度下降。

1.  我们为什么需要动量？

1.  NAG 背后的动机是什么？

1.  Adagrad 如何自适应地设置学习率？

1.  Adadelta 的更新规则是什么？

1.  RMSProp 如何克服 Adagrad 的局限性？

1.  定义 Adam 的更新方程。

# 进一步阅读

更多信息，请参考以下链接：

+   *用于在线学习和随机优化的自适应次梯度方法*，作者：John Duchi 等，[`www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf`](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

+   *Adadelta：一种自适应学习率方法*，作者：Matthew D. Zeiler，[`arxiv.org/pdf/1212.5701.pdf`](https://arxiv.org/pdf/1212.5701.pdf)

+   *Adam：一种用于随机优化的方法*，作者：Diederik P. Kingma 和 Jimmy Lei Ba，[`arxiv.org/pdf/1412.6980.pdf`](https://arxiv.org/pdf/1412.6980.pdf)

+   *Adam 及其扩展算法的收敛性研究*，作者：Sashank J. Reddi, Satyen Kale, 和 Sanjiv Kumar，[`openreview.net/pdf?id=ryQu7f-RZ`](https://openreview.net/pdf?id=ryQu7f-RZ)

+   *将 Nesterov 动量整合到 Adam 中*，作者：Timothy Dozat，[`cs229.stanford.edu/proj2015/054_report.pdf`](http://cs229.stanford.edu/proj2015/054_report.pdf)
