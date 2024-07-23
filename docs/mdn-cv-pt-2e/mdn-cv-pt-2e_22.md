# 第十八章：将模型部署到生产环境

将模型部署到生产环境是促使外部方使用我们模型的一步。我们应该向世界公开我们的模型，并开始对真实世界未见过的输入进行预测。

部署训练过的 PyTorch 模型并不足以实现部署。我们需要额外的服务器组件，以建立从现实世界到 PyTorch 模型的通信渠道，以及从模型返回到现实世界的通信渠道。重要的是，我们要知道如何创建 API（用户可以通过该接口与模型交互），将其封装为独立的应用程序（以便可以部署在任何计算机上），并将其部署到云端 - 这样任何具有必要 URL 和凭据的人都可以与模型交互。为了成功将模型移至生产环境，所有这些步骤都是必需的。

此外，我们还必须处理预测延迟（获取预测所需的时间）和模型大小（在部署在像移动设备/手表之类的边缘设备时）的限制，而不会影响准确性。

此外，作为生产支持的一部分，我们将不断关注模型部署后输入的情况，然后检查在训练模型时使用的图像与在实际部署过程中输入的图像之间是否存在显著的漂移（偏差）。

在本章中，我们将部署一个简单的应用程序，并建立一个识别输入漂移的机制。

本章将涵盖以下主题：

+   了解 API 的基础知识

+   在本地服务器上创建 API 并进行预测

+   将模型容器化并部署到云端

+   识别数据漂移

+   使用**Facebook AI Similarity Search**（**FAISS**）构建向量存储库

本章中的所有代码片段均可在 GitHub 存储库的 `Chapter18` 文件夹中找到：[`bit.ly/mcvp-2e`](https://bit.ly/mcvp-2e)。

此外，我们的 GitHub 存储库中还涵盖以下主题：

+   将模型转换为**Open Neural Network Exchange**（**ONNX**）格式

随着领域的发展，我们将定期向 GitHub 存储库添加有价值的补充内容。请查看每章节目录中的 `supplementary_sections` 文件夹，以获取新的和有用的内容。

需要注意的是部署模型的典型工作流程如下：

1.  在本地服务器上创建 API 并进行预测

1.  将应用程序容器化

1.  在云端部署 Docker 容器

1.  建立训练图像的向量存储库

1.  识别现实世界图像中的偏差（漂移），以便我们知道是否需要对模型进行微调

# 了解 API 的基础知识

到目前为止，我们知道如何为各种任务创建深度学习模型。它接受/返回张量作为输入/输出。但是像客户/最终用户这样的外部人员只会用图像和类别来谈论。此外，他们希望通过可能与 Python 无关的渠道发送和接收输入/输出。互联网是最容易进行通信的渠道。因此，对于客户端来说，最佳部署场景是如果我们能够设置一个公开可用的 URL，并要求他们将他们的图像上传到那里。这种范例称为**应用程序编程接口**（**API**），它具有标准协议，接受输入并在互联网上传输输出，同时将用户从输入处理或输出生成的细节中抽象出来。

API 中一些常见的协议包括 `POST`、`GET`、`PUT` 和 `DELETE`，这些协议由客户端作为请求发送到主机服务器，并附带相关数据。基于请求和数据，服务器执行相关任务，并以适当的数据形式返回**响应**，客户端可以在其下游任务中使用。在我们的案例中，客户端将发送一个带有图像作为文件附件的 POST 请求。我们应该保存文件，处理它，并返回适当的类作为请求的响应，这样我们的工作就完成了。

请求是通过互联网发送的组织数据包，用于与 API 服务器通信。通常，请求的组成部分如下：

+   **终端点 URL**：这是 API 服务的地址。例如，[`www.packtpub.com/`](https://www.packtpub.com/) 就是连接到 Packt Publishing 服务并浏览其最新图书目录的终端点。

+   **一系列标头**：这些信息帮助 API 服务器返回输出；例如，如果标头包含客户端使用移动设备，则 API 可以返回一个移动友好的 HTML 页面布局。

+   **一系列查询**：这确保只从服务器数据库中获取相关的项目。例如，搜索字符串 `PyTorch` 将仅返回前面示例中与 PyTorch 相关的书籍。请注意，在本章中，我们不会处理查询，因为对图像的预测不需要查询，而是需要文件名。

+   **一系列文件**：这些文件可以上传到服务器，或者在我们的情况下，用于进行深度学习预测。

cURL 是一个计算机软件项目，提供了一个库和命令行工具，用于使用各种网络协议传输数据。它是最轻量级、常用和简单的应用程序之一，用于调用 API 请求并获取响应。

为了说明这一点，我们将在接下来的章节中实现一个名为 FastAPI 的现成 Python 模块，使我们能够做到以下几点：

1.  设置通信 URL。

1.  在发送到 URL 时，接受来自各种环境/格式的输入。

1.  将每种输入形式转换为机器学习模型需要的确切格式。

1.  使用训练好的基于深度学习的模型进行预测。

1.  将预测转换为正确的格式，并通过预测响应客户端的请求。

我们将使用**表面缺陷数据集**（**SDD**）（[`www.vicos.si/resources/kolektorsdd/`](https://www.vicos.si/resources/kolektorsdd/)）作为示例来演示这些概念。

在了解基本设置和代码后，您可以为任何类型的深度学习任务创建 API，并通过本地机器上的 URL 提供预测。下一个逻辑步骤是将应用程序容器化，并将其部署到云上，以便从世界任何地方访问应用程序。部署的应用程序随后将需要支持，因为重要的是我们了解如何识别现实世界数据中的偏差，如果模型表现不佳的话。

在接下来的几节中，我们将学习如何将应用程序封装在一个自包含的 Docker 镜像中，该镜像可以在云中任何地方进行部署和运行。让我们使用 FastAPI，一个 Python 库，来创建 API，并验证我们可以直接从终端进行预测（而不使用 Jupyter 笔记本）。

# 创建一个 API，并在本地服务器上进行预测。

在本节中，我们将学习如何在本地服务器上进行预测（与云无关）。从高层次来看，这包括以下步骤：

1.  安装 FastAPI

1.  创建一个接受传入请求的路由。

1.  将传入请求保存到磁盘上。

1.  加载请求的图像，然后进行预处理并使用训练好的模型进行预测。

1.  对结果进行后处理，并将预测发送回相同的传入请求作为响应。

本节中的所有步骤都被总结为一个视频教程：[`tinyurl.com/MCVP-Model2FastAPI`](https://tinyurl.com/MCVP-Model2FastAPI)。

让我们开始安装 FastAPI。

## 安装 API 模块和依赖项。

由于 FastAPI 是一个 Python 模块，我们可以使用`pip`进行安装，并准备编写一个 API。我们现在将打开一个新终端，并运行以下命令：

```py
$pip install fastapi uvicorn aiofiles jinja2 
```

我们已安装了几个与 FastAPI 一起需要的依赖项。`uvicorn`是一个用于设置 API 的最小低级服务器/应用程序接口。`aiofiles`使服务器可以异步处理请求，例如同时接受和响应多个独立的并行请求。这两个模块是 FastAPI 的依赖项，我们不会直接与它们交互。

让我们创建所需的文件，并在下一节中编写它们的代码。

## 提供一个图像分类器。

在本节中，我们将学习如何在本地部署模型，以便从端点接收预测。

第一步是设置文件夹结构，如下所示：

![一个黑白 Logo  自动生成描述](img/B18457_18_01.png)

图 18.1：模型服务的文件夹结构

设置非常简单，如下所示：

+   `files`文件夹将作为传入请求的下载位置。

+   `sdd.weights.pth`包含我们训练的 SDD 模型的权重。

+   `sdd.py`将包含加载权重、接受传入图像、预处理、预测和后处理预测的逻辑。

+   `server.py`将包含 FastAPI 功能，可以设置 URL，从 URL 接受客户端请求，发送/接收来自`sdd.py`的输入/输出，并将输出作为响应发送给客户端请求。

注意

`files`文件夹为空，仅用于存储上传的文件。

我们假设已经有了训练模型的权重作为`sdd.weights.pth`。

训练过程类似于我们在*第四章*中进行图像分类的训练。

与训练模型相关联的笔记本位于书的 GitHub 存储库的`Chapter18`文件夹中的第一部分的`vector_stores.ipynb`笔记本中。

现在让我们理解`ssd.py`和`server.py`的构成并编写它们。

### `sdd.py`

如前所述，`sdd.py`文件应该具有加载模型并返回给定图像的预测的逻辑。

我们已经熟悉如何创建 PyTorch 模型。类的唯一额外组件是`predict`方法，用于对图像进行必要的预处理和结果的后处理。您可以按照以下步骤操作：

以下代码可在书的 GitHub 存储库的`Chapter18`文件夹中的`ssd.py`中找到，网址为`https://bit.ly/mcvp-2e`。

1.  我们首先创建`model`类，该类构成了模型的架构：

    ```py
    from torch_snippets import *
    class SDD(nn.Module):
      classes = ['defect','non_defect']
      def __init__(self, model, device='cpu'):
        super().__init__()
        self.model = model.to(device)
        self.device = device 
    ```

1.  以下代码块突出显示了前向方法：

    ```py
     @torch.no_grad()
      def forward(self, x):
        x = x.view(-1,3,224,224).to(device)
        pred = self.model(x)
        conf = pred[0][0]
        clss = np.where(conf.item()<0.5,'non_defect','defect')
        print(clss)
        return clss.item() 
    ```

在上述代码中，我们首先重塑输入图像，通过模型，并获取与图像对应的预测类。

以下代码块突出显示了`predict`方法，用于进行必要的预处理和后处理：

1.  ```py
     def predict(self, image):
        im = (image[:,:,::-1])
        im = cv2.resize(im, (224,224))
        im = torch.tensor(im/255)
        im = im.permute(2,0,1).float()
        clss = self.forward(im)
        return {"class": clss} 
    ```

总之，在前面的步骤中，在`__init__`方法中，我们初始化模型（其中在上一步中加载了预训练权重）。在`forward`方法中，我们通过模型传递图像并获取预测。在`predict`方法中，我们从预定义路径加载图像，预处理图像，然后通过模型的`forward`方法传递它，并在返回预测类时将输出包装在字典中。

### server.py

这是 API 中连接用户请求与 PyTorch 模型的代码部分。让我们逐步创建文件：

以下代码可在书的 GitHub 存储库的`Chapter18`文件夹中的`server.py`中找到，网址为`https://bit.ly/mcvp-2e`。

1.  加载库：

    ```py
    import os, io
    from sdd import SDD
    from PIL import Image
    from fastapi import FastAPI, Request, File, UploadFile 
    ```

`FastAPI`是基本的服务器类，用于创建 API。

`请求`、`文件`和`UploadFile`是客户端请求和它们将上传的文件的代理占位符。有关更多详细信息，请参阅官方 FastAPI 文档：[`fastapi.tiangolo.com/`](https://fastapi.tiangolo.com/)。

1.  加载模型：

    ```py
    # Load the model from sdd.py
    device = 'cuda' if torch.cuda.is_availabe() else 'cpu'
    model = SDD(torch.load('sdd.weights.pth', map_location=device) 
    ```

1.  创建一个`app`模型，它可以提供一个用于上传和显示的 URL：

    ```py
    app = FastAPI() 
    ```

1.  创建一个`"/predict"`的 URL，以便客户端可以向`"<hosturl>/predict"`发送`POST`请求并接收响应（我们将在下一节学习有关`<hosturl>`的信息，即服务器的信息）：

    ```py
    @app.post("/predict")
    def predict(request: Request, file:UploadFile=File(...)):
        content = file.file.read()
        image = Image.open(io.BytesIO(content)).convert('L')
        output = model.predict(image)
        return output 
    ```

这就是全部！我们有所有的组件来利用我们的图像分类器在我们的本地服务器上进行预测。让我们设置服务器并在本地服务器上进行一些预测。

### 运行服务器

现在我们已经设置好所有的组件，可以运行服务器了。打开一个新终端并`cd`到包含`sdd.py`、`server.py`的文件夹：

1.  运行服务器：

    ```py
    $ uvicorn server:app 
    ```

您将看到如下消息：

![计算机屏幕截图的描述 自动生成](img/B18457_18_02.png)

图 18.2：应用程序启动期间的消息

`Uvicorn running on ...` 消息表明服务器已启动并运行。

1.  要获取预测结果，我们将在一个新的终端中运行以下命令，为`/home/me/Pictures/defect.png`中存在的样本图像获取预测：

    ```py
    $ curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/home/me/Pictures/defect.png;type=image/png" 
    ```

上述代码行的主要组件如下：

+   **REST API 方法**：所使用的方法是 POST，这表示我们希望将自己的数据发送到服务器。

+   **URL – 服务器地址**：服务器主机 URL 是`http://127.0.0.1:8000/`（这是本地服务器，使用`8000`作为默认端口），`/predict/`是提供给客户端创建`POST`请求的路由；未来客户端必须将其数据上传到 URL`http://127.0.0.1:8000/predict`。

+   **头部**：请求包含`-H`标志形式的组件。这些附加信息，例如以下内容：

    1.  输入内容类型将是什么 - `multipart/form-data` - 这是 API 术语，表示输入数据以文件的形式提交。

    1.  预期的输出类型是什么 - `application/json` - 这意味着 JSON 格式。还有其他格式，如 XML、文本和`octet-stream`，这些根据生成的输出复杂性而应用。

+   **文件**：最终的`-F`标志指向要上传的文件所在的位置及其类型。

一旦运行了上述代码，输出字典将被打印在终端中。

现在我们可以从本地服务器获取模型预测结果了。

现在我们已经了解了如何构建服务器，在接下来的部分中，我们将学习如何将应用程序容器化，以便可以在任何系统上运行。

# 容器化应用程序

我们宁愿安装一个具有所有内容的包，而不是安装多个单独的包（如运行应用程序所需的单独模块和代码），然后稍后再将它们连接起来。因此，我们需要学习如何使用 Docker，它本质上是一个带有代码的压缩操作系统。创建的 Docker 容器是轻量级的，将仅执行我们想要执行的任务。在我们的示例中，我们将创建的 Docker 镜像将运行用于预测 SDD 图像类别的 API。但首先，让我们了解一些 Docker 行话。

**Docker 镜像** 是打包代码及其所有依赖关系的标准软件单元。这样，应用程序可以从一个计算环境快速可靠地运行到另一个计算环境。Docker 镜像是一个轻量级、独立运行的软件包，包含运行应用程序所需的一切：代码、运行时、系统工具、系统库和设置。

**Docker 容器** 是镜像的快照，将在需要部署的地方实例化。我们可以从单个镜像创建任意数量的 Docker 容器，它们将执行相同的任务。可以将镜像视为蓝图，将容器视为从该蓝图创建的实例。

在高层次上，我们将学习如何执行以下任务：

1.  创建一个 Docker 镜像。

1.  将其创建为 Docker 容器并进行测试。

让我们从创建一个 Docker 容器开始。

## 构建 Docker 镜像

在前一节中，我们构建了一个 API，该 API 在本地服务器上接收图像并返回图像的类别及其概率。现在，是时候将我们的 API 封装成一个可以在任何地方部署的软件包了。

确保您的计算机安装了 Docker。您可以参考[`docs.docker.com/get-docker/`](https://docs.docker.com/get-docker/)中的安装说明。

创建 Docker 容器的过程包括三个步骤：

1.  创建一个 `requirements.txt` 文件。

1.  创建一个 Dockerfile。

1.  构建 Docker 镜像并创建 Docker 容器。

以下各节的代码还可以在这里进行视频演示：[`tinyurl.com/MCVP-2E-model2fastapi-docker`](https://tinyurl.com/MCVP-2E-model2fastapi-docker)。

现在，我们将详细了解并理解这四个步骤，接下来，我们将学习如何将镜像发送到 AWS 服务器。

### 创建 `requirements.txt` 文件

我们需要告诉 Docker 镜像安装哪些 Python 模块以运行应用程序。`requirements.txt` 文件包含所有这些 Python 模块的列表：

1.  打开一个终端，进入包含 `sdd.py`、`server.py` 的文件夹。接下来，我们将在本地终端的 `root` 文件夹中创建一个空白虚拟环境并激活它：

    ```py
    $ python3 -m venv fastapi-venv
    $ source fastapi-env/bin/activate 
    ```

我们创建空白虚拟环境的原因是确保*只*在环境中安装所需的模块，以便在发布时不浪费宝贵的空间。

1.  安装所需的包（`fastapi`、`uvicorn`、`aiofiles`、`torch` 和 `torch_snippets`）以运行 SDD 应用程序：

    ```py
    $ pip install fastapi uvicorn aiofiles torch torch_snippets 
    ```

1.  在同一个终端中，运行以下命令以安装所有必需的 Python 模块：

    ```py
    $ pip freeze > requirements.txt 
    ```

上述代码将所有 Python 模块及其相应的版本号提取到 `requirements.txt` 文件中，这将用于在 Docker 镜像中安装依赖项。

现在我们已经具备了所有的先决条件，让我们在下一节中创建 Dockerfile。

### 创建 Dockerfile

如前所述，Docker 镜像是一个独立的应用程序，完全具备自己的操作系统和依赖项。给定一个计算平台（例如 EC2 实例），镜像可以独立运行并执行其设计的任务。为此，我们需要为 Docker 应用程序提供必要的指令 - 依赖项、代码和命令。

让我们在 SDD 项目的 `root` 目录中创建一个名为 `Dockerfile` 的文本文件，其中包含 `server.py`、`sdd.py`（我们已经在创建项目文件夹后放置了）。文件需要命名为 `Dockerfile`（无扩展名），作为约定。文本文件的内容如下：

以下代码位于书籍 GitHub 仓库的 `Chapter18` 文件夹中的 `Dockerfile` 中，网址为 [`bit.ly/mcvp-2e`](https://bit.ly/mcvp-2e)。

```py
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY . /app
EXPOSE 5000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0"] 
```

让我们逐步理解上述代码：

1.  `FROM` 指示我们要使用哪个操作系统基础。`tiangolo/uvicorn-gunicorn-fastapi:python3.7` 的位置是一个地址，由 Docker 从互联网解析，它获取一个已经安装了 Python 和其他 FastAPI 模块的基础镜像。

1.  接下来，我们复制我们创建的 `requirements.txt` 文件。这提供了我们想要安装的包。在下一行中，我们要求镜像使用 `pip install` 来安装包。

1.  `WORKDIR` 是我们的应用程序将运行的文件夹。因此，我们在 Docker 镜像中创建一个名为 `/app` 的新文件夹，并将 `root` 文件夹的内容复制到镜像的 `/app` 文件夹中。

1.  最后，像前一节那样运行服务器。

这样，我们就设置了一个蓝图，可以从头开始创建一个全新的操作系统和文件系统（可以想象成一个新的可安装的 Windows 光盘），其中只包含我们指定的代码并且只运行一个应用程序，即 FastAPI。

### 构建 Docker 镜像和创建 Docker 容器

注意，到目前为止，我们只创建了 Docker 镜像的蓝图。让我们构建镜像并创建一个容器。

提供了构建 Dockerfile 过程的视频演示，链接在 YouTube 上：[`bit.ly/MCVP-Docker-Deployment`](https://bit.ly/MCVP-Docker-Deployment)。

在同一个终端中（包含应用程序文件的根目录），从这里运行以下命令：

1.  构建 Docker 镜像并将其标记为 `sdd:latest`：

    ```py
    $ docker build -t sdd:latest . 
    ```

经过一长串输出后，我们得到以下内容，告诉我们镜像已构建完成：

![](img/B18457_18_05.png)

图 18.3：创建 Docker 镜像

我们已成功创建了一个名为 `sdd:latest` 的 Docker 镜像（其中 `sdd` 是镜像名称，`latest` 是我们给出的标签，表示其版本号）。Docker 在系统中维护一个注册表，所有这些镜像都可以从中访问。这个 Docker 注册表现在包含一个独立的镜像，其中包含运行 SDD API 所需的所有代码和逻辑。

我们可以通过在命令提示符中输入 `$ docker image ls` 来随时检查 Docker 注册表。

1.  使用 `-p 5000:5000` 启动构建好的镜像，在镜像内部将端口 `5000` 转发到本地机器的 `5000` 端口。最后一个参数是从镜像创建的容器的名称：

    ```py
    $ docker run -p 5000:5000 sdd:latest 
    ```

端口转发非常重要。通常，我们无法控制云暴露哪些端口。因此，尽管我们的 `uvicorn` 模型为 `POST` 操作创建了 `5000` 端口，作为演示，我们仍然使用 Docker 的功能将外部请求从 `5000` 端口路由到 `5000` 端口，这是 `uvicorn` 监听的位置。

这应该会显示最后几行的提示，如下所示：

![](img/B18457_18_06.png)

图 18.4：应用程序启动过程中的消息

1.  现在，从新终端运行 `curl` 请求并访问 API，如前文所述，但这次应用程序是通过 Docker 提供的：

![](img/B18457_18_07.png)

图 18.5：执行预测请求

尽管到目前为止我们还没有将任何东西移至云端，但将 API 包装在 Docker 中可以避免我们再次担心 `pip install` 或复制粘贴代码的问题。

现在我们已经了解了如何将应用程序容器化，让我们继续，在接下来的部分中将 Docker 容器在云上部署和运行。

# 在云上运行和部署 Docker 容器

我们将依赖 AWS 来满足我们的云需求。我们将使用 AWS 的两个免费服务来达到我们的目标：

+   **弹性容器注册表** (**ECR**)：在这里，我们将存储我们的 Docker 镜像。

+   **EC2**：在这里，我们将创建一个 Linux 系统来运行我们的 API Docker 镜像。

在本节中，我们将仅关注 ECR。以下是将 Docker 镜像推送到云的步骤的高级概述：

1.  在本地机器上配置 AWS。

1.  在 AWS ECR 上创建一个 Docker 仓库并推送 `sdd:latest` 镜像。

1.  创建一个 EC2 实例。

1.  在 EC2 实例上安装依赖项。

1.  在 EC2 实例上创建并运行推送的 Docker 镜像 *第二步*。

下面各节的代码摘要也可以在视频演示中找到：[`tinyurl.com/MCVP-FastAPI2AWS`](https://tinyurl.com/MCVP-FastAPI2AWS)。

让我们从下一节开始实施上述步骤，首先配置 AWS。

## 配置 AWS

我们将从命令提示符登录到 AWS 并推送我们的 Docker 镜像。让我们逐步进行：

1.  在 [`aws.amazon.com/`](https://aws.amazon.com/) 创建一个 AWS 账户并登录。

1.  在本地安装 AWS CLI（其中包含 Docker 镜像）。

1.  通过在本地终端中运行 `aws --version` 来验证是否已安装。

1.  在终端中运行 `aws configure`，在询问时提供适当的凭据。这些凭据可以在 IAM 部分找到：

    ```py
    $ aws configure
    AWS Access Key ID [None]: **AKIAIOSFODNN7EXAMPLE**
    AWS Secret Access Key [None]:**wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY**
    Default region name [None]: **region**
    Default output format [None]:**json** 
    ```

我们现在已经从我们的计算机登录到了 Amazon 的服务。在接下来的部分，让我们连接到 ECR 并推送 Docker 镜像。

## 创建 AWS ECR 上的 Docker 仓库并推送镜像

现在，我们将创建 Docker 仓库，如下所示：

1.  配置后，使用以下命令登录到 AWS ECR（以下代码都在一行上），其中需要提供 `region` 和您的 `aws_accound_id`，如下面代码中的粗体所示（请记住在每一步中使用您自己的这些变量值）：

    ```py
    $ aws ecr get-login-password --region **region****aws_account_id.****region** | docker login --username AWS --password-stdin dkr.ecr..amazonaws.com 
    ```

上述代码行在亚马逊云中创建并连接您自己的 Docker 注册表。

1.  通过运行以下命令在 CLI 中创建仓库：

    ```py
    $ aws ecr create-repository --repository-name sdd_app 
    ```

上述代码创建了一个可以存放您的 Docker 镜像的云端位置。

通过运行以下命令为本地镜像打标签，以便在推送镜像时将其推送到已标记的仓库：

1.  ```py
    $ docker tag sdd:latest **aws_account_id****region**.dkr.ecr..amazonaws.com/sdd_app 
    ```

1.  运行以下命令将本地 Docker 镜像推送到云中的 AWS 仓库：

    ```py
    $ docker push **aws_account_id****region**.dkr.ecr..amazonaws.com/sdd_app 
    ```

我们已成功在云中为我们的 API 创建了一个位置，并将 Docker 镜像推送到了该位置。正如您现在所知，这个镜像已经具备了运行 API 所需的所有组件。唯一剩下的就是在云中将其创建成 Docker 容器，这样我们就成功地将我们的应用程序推向了生产环境！

1.  现在，让我们继续创建一个 Amazon Linux 2 AMI - `t2.micro` 实例，带有 20 GB 的空间。在创建实例时，在 `配置安全组` 部分添加 `允许 HTTP 流量规则`，以便我们将要部署的应用程序可以从任何地方访问。

1.  复制看起来像这样的 EC2 实例名称：

    ```py
    ec2-18-221-11-226.us-east-2.compute.amazonaws.com 
    ```

1.  使用以下命令在本地终端登录到 EC2 实例：

    ```py
    $ ssh ec2-user@ec2-18-221-11-226.us-east-2.compute.amazonaws.com 
    ```

接下来，让我们安装在 EC2 机器上运行 Docker 镜像所需的依赖项，然后我们就可以运行 API 了。

## 拉取镜像并构建 Docker 容器

所有以下命令都需要在我们之前登录到的 EC2 控制台中运行：

1.  在 Linux 机器上安装和配置 Docker 镜像：

    ```py
    $ sudo yum install -y docker
    $ sudo groupadd docker
    $ sudo gpasswd -a ${USER} docker
    $ sudo service docker restart 
    ```

`groupadd` 和 `gpasswd` 确保 Docker 具有运行所需的所有权限。

1.  如您之前所做的那样，在 EC2 实例中配置 AWS，并重新启动机器：

    ```py
    $ aws configure
    AWS Access Key ID [None]: **AKIAIOSFODNN7EXAMPLE**
    AWS Secret Access Key [None]:**wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY**
    Default region name [None]: **us-west-****2**
    Default output format [None]:**json**
    $ reboot 
    ```

1.  使用以下命令从本地终端重新登录到实例：

    ```py
    $ ssh ec2-user@**ec2-****18****-****221****-****11****-****226****.us-east-****2****.compute.amazonaws.com** 
    ```

1.  现在，从已登录（已安装 Docker）的 EC2 控制台（其中已登录到 AWS ECR），登录 AWS ECR（根据以下命令中显示的粗体区域更改区域）：

    ```py
    $ aws ecr get-login-password --region **region****aws_account_id.****region** | docker login --username AWS --password-stdin dkr.ecr..amazonaws.com 
    ```

1.  从 AWS ECR 拉取 Docker 镜像：

    ```py
    $ docker pull **aws_account_id****region**.dkr.ecr..amazonaws.com/sdd_app:latest 
    ```

最后，在 EC2 机器上运行已拉取的 Docker 镜像：

1.  ```py
    docker run -p 5000:5000 **aws_account_id****region**.dkr.ecr..amazonaws.com/sdd_app 
    ```

我们在 EC2 上运行我们的 API。我们只需获取机器的公共 IP 地址，并用该地址替换`127.0.0.1`运行`curl`请求。

1.  现在您可以从任何计算机发起`POST`请求，EC2 实例将对其进行响应，并为我们上传的衣物图像提供预测：

    ```py
    $ curl -X POST "http://54.229.16.169:5000/predict" -H "accept: application/ json" -H "Content-Type: multipart/form-data" -F "file=@/home/me/Pictures/ defect.png;type=image/png" 
    ```

上述代码会产生以下输出：

1.  ```py
    {"class":"non-defect","confidence":"0.6488"} 
    ```

在本节中，我们已经能够为 EC2 安装依赖项，拉取 Docker 镜像，并运行 Docker 容器，以便任何具有 URL 的用户可以对新图像进行预测。

整个过程的概述可以在此视频演示中查看：`https://tinyurl.com/MCVP-FastAPI2AWS`。

可以在关联的 GitHub 存储库的`Chapter18`文件夹中的 PDF 文件中找到具有屏幕截图的逐步说明，标题为`Shipping and running the Docker container on cloud`。

现在我们了解了如何部署模型，接下来我们将学习如何识别在实际输入数据（在真实世界中）与训练时使用的数据相比时处于分布之外的场景。

# 识别数据漂移

在典型的表格数据集中，通过查看模型训练的数据集的摘要统计信息，可以相对容易地理解传入数据点是否为异常值。然而，计算机视觉模型并不那么直接 - 我们已经在*第四章*中看到它们的怪癖，在那里，仅仅通过将像素平移几个像素，预测的类别就发生了变化。然而，在图像的情况下，这并不是唯一的数据漂移场景。进入生产模型的数据与模型训练所用的数据有很多不同之处。这些可能是显而易见的事情，比如图像的照明不对或者图像中预期的主题错误，也可能是人眼看不见的微妙之处。

在本节中，我们将了解如何实时测量输入图像（用于预测的真实世界图像）与模型训练期间使用的图像之间的漂移方式。

我们测量漂移的原因如下：

+   如果真实世界的输入图像不像训练时使用的图像，那么我们可能会错误地进行预测。

+   此外，我们希望在尽早实施模型后尽快获得反馈，以便我们可以重新训练/微调模型。

我们将采用以下策略来确定新图像是否为分布之外：

1.  从中间层提取嵌入（例如，`avgpool`层）。

1.  对训练图像和来自真实世界的任何新图像执行上述步骤：

1.  比较新图像的嵌入与所有训练图像的嵌入：

1.  您还可以尝试仅使用与新图像同一类别的训练图像的嵌入进行实验。

1.  如果新图像的嵌入与训练图像的嵌入之间的距离较大，则新图像属于分布之外。

我们将在代码中实现上述内容如下：

以下代码位于书籍 GitHub 存储库的`Chapter18`文件夹中的`measuring_drift.ipynb`中，网址为`https://bit.ly/mcvp-2e`。

1.  获取包含训练模型代码的存储库：

    ```py
    %%capture
    try:
      from torch_snippets import *
    except:
      %pip install torch-snippets gitPython lovely-tensors
      from torch_snippets import *
    from git import Repo
    repository_url = 'https://github.com/sizhky/quantization'
    destination_directory = '/content/quantization'
    if exists(destination_directory):
      repo = Repo(destination_directory)
    else:
      repo = Repo.clone_from(repository_url, destination_directory)
    %cd {destination_directory}
    %pip install -qq -r requirements.txt 
    ```

1.  训练模型：

    ```py
    # Change to `Debug=false` in the line below
    # to train on a larger dataset
    %env DEBUG=true
    !make train 
    ```

1.  获取训练集、验证集以及数据加载器：

    ```py
    from torch_snippets import *
    from src.defect_classification.train import get_datasets, get_dataloaders
    trn_ds, val_ds = get_datasets(DEBUG=True)
    trn_dl, val_dl = get_dataloaders(trn_ds, val_ds) 
    ```

1.  加载模型：

    ```py
    model = torch.load('model.pth').cuda().eval() 
    ```

1.  获取训练过程中使用的图像的嵌入：

    ```py
    results = []
    for ix, batch in enumerate(iter(trn_dl)):
      inter = model.avgpool(model.features(batch[0].cuda()))[:,:,0,0].\
                                                    detach().cpu().numpy()
      results.append(inter)
    results = np.array(results)
    results = results.reshape(-1, 512) 
    ```

1.  获取样本验证图像的嵌入，并计算与训练图像的嵌入之间的距离：

    ```py
    im = val_ds[0]['image'][None].cuda()
    tmp = np.array(model.avgpool(model.features(im))[0,:,0,0].detach().cpu().\
                                                             numpy())
    dists1 = np.sum(np.abs(results - tmp), axis=1) 
    ```

1.  现在，让我们使用一个完全无关的图像进行相同的练习：

    ```py
    !wget https://as2.ftcdn.net/v2/jpg/01/42/16/37/1000_F_142163797_YxZaY95j5ckLgb6KoM5KC11Eh9QiZsYx.jpg -O /content/sample_defects.jpg
    im = (cv2.imread(path)[:,:,::-1])
    im = cv2.resize(im, (224,224))
    im = torch.tensor(im/255)
    im = im.permute(2,0,1).float().to(device)
    im = im.view(1,3,224,224)
    tmp = np.array(model.avgpool(model.features(im))[0,:,0,0].detach().cpu().\
                                                             numpy())
    dists2 = np.sum(np.abs(results - tmp), axis=1) 
    ```

1.  让我们绘制训练图像与验证图像以及无关（随机）图像之间距离分布：

    ```py
    import seaborn as sns
    df = pd.DataFrame(
        {'distance': np.r_[dists1, dists2],
         'source': ['val image']*len(dists1)+['random image']*len(dists2)})
    # Just switch x and y
    sns.boxplot(y=df["source"], x=df["distance"]) 
    ```

![一个图表的图解](img/B18457_18_08.png)

图 18.6：训练图像与验证图像以及随机图像之间的距离分布

从上述内容可以看出，与来自分布内部的图像的距离范围相比，随机图像的距离范围差异极大。通过这种方式，我们可以了解我们正在预测的图像是否落在训练图像的分布范围内。此外，当超过阈值的现实世界图像数量（或百分比）超出分布时，考虑重新训练模型是值得的。

在本节中，我们计算了预测图像与所有训练图像之间的距离。在这种情况下，由于训练图像的数量有限（几百个），这变得更加容易。

如果训练图像数量达到约一百万？我们如何快速执行距离计算？在接下来的部分中，我们将学习向量存储是如何帮助解决这个问题的。

# 使用向量存储

向量存储的直觉是：如果我们可以将所有向量分组到某些簇中，对于新向量，我们可以首先确定它可能属于的簇，然后计算新向量与属于同一簇的图像之间的距离。

这个过程有助于避免跨所有图像的计算，从而大大减少计算时间。

FAISS 是由 Meta 构建的开源库，用于在向量之间进行快速的近似相似性搜索。有广泛的开源和专有向量存储库。我们强烈建议您在理解向量存储的需求后，查看这些库。

现在我们了解了向量存储，让我们继续执行以下步骤：

1.  将训练图像嵌入存储在向量存储中。

1.  比较检索查询（验证/实际）图像的三个最接近图像所需的时间，即：

    1.  没有使用向量存储

    1.  我们使用向量存储

1.  增加训练图像数量，然后比较在前述两种情况下检索给定图像最接近图像所需的时间。

让我们了解如何编写代码：

以下代码作为 `Chapter18` 文件夹中的 `vector_stores.ipynb` 在 GitHub 仓库 [`bit.ly/mcvp-2e`](https://bit.ly/mcvp-2e) 中提供。

1.  安装 FAISS 库：

    ```py
    !pip install faiss-gpu 
    ```

在前一节（关于数据漂移）的前五个步骤保持不变——即获取训练和验证数据集，训练和加载模型，并获取训练图像的嵌入。

1.  对所有训练图像的向量进行索引：

    ```py
    import faiss
    import numpy as np
    index = faiss.IndexFlatL2(results.shape[1])  # L2 distance
    index.add(results)
    faiss.write_index(index, "index_file.index") 
    ```

在上述代码中，我们获取训练嵌入（结果）并使用 `IndexFlatL2` 方法进行索引。接下来，我们将索引写入磁盘。

1.  获取新图像的向量：

    ```py
    im = val_ds[0]['image'][None].cuda()
    tmp = np.array(model.avgpool(model.features(im))[0,:,0,0].detach().cpu().\
                                                             numpy())
    query_vector = tmp.reshape(1,512).astype('float32') 
    ```

1.  查找最相似的向量：

    ```py
    %%time
    k = 3  # Number of nearest neighbors to retrieve
    D, I = index.search(query_vector.astype('float32'), k) 
    ```

在上述代码中，`D` 表示距离，`I` 表示与查询图像最相似的图像的索引。我们可以看到，搜索所有图像花费了 0.2 毫秒。

1.  大幅增加训练图像向量的数量：

    ```py
    vectors = np.array(results.tolist()*10000, dtype=np.float32)
    print(vectors.shape)
    index = faiss.IndexFlatL2(vectors.shape[1])  # L2 distance
    index.add(vectors)
    faiss.write_index(index, "index_file_960k.index") 
    ```

在上述代码中，我们通过重复训练图像列表 10000 次来人为增加训练图像嵌入的数量。然后，我们对嵌入进行索引并写入磁盘。

1.  计算搜索与查询向量最接近的向量所需的时间：

    ```py
    %%time
    k = 3  # Number of nearest neighbors to retrieve
    D, I = index.search(query_vector.astype('float32'), k) 
    ```

请注意，在前述情况下，获取相似向量需要约 700 毫秒。

1.  计算在没有索引的情况下检索三个最接近向量所需的时间：

    ```py
    %%time
    distances = np.sum(np.square(query_vector - vectors), axis=1)
    sorted_distances = np.sort(distances) 
    ```

请注意，在没有索引的情况下，获取最接近向量需要约 5 秒钟。

在以下图表中，我们将看到在两种情况下（索引和非索引）随着训练图像数量增加，查找给定图像的最接近训练图像所需的时间：

![一幅具有线条和蓝线的图表，说明自动生成](img/B18457_18_09.png)

图 18.7：在增加训练图像数量时，比较带有索引和不带索引时识别最接近匹配图像所需的时间。

注意，有索引与无索引之间的差异随着训练图像数量的增加呈指数级增长。尽管如此，在信息检索对企业至关重要的世界中，以最快的时间获得解决方案是重要的，而向量存储是节省时间的关键方式。

# 总结

在本章中，我们了解了将模型部署到生产环境中所需的额外步骤。我们学习了 API 是什么及其组成部分。在使用 FastAPI 创建 API 后，我们简要介绍了创建 API Docker 镜像、创建 Docker 容器并将其推送到云端的核心步骤。然后，我们学习了如何识别与原始数据集不符的图像的方法。此外，我们还学习了如何利用 FAISS 更快地计算与相似向量的距离。

总结一下，我们已经看到了将模型部署到生产环境中的所有步骤，包括构建 Docker 容器、在 AWS 云上部署、识别数据漂移以及了解何时重新训练模型的场景，并通过更少的计算快速执行图像相似性。

图像是迷人的。存储图像是人类最早的努力之一，也是捕捉内容最强大的方式之一。在 21 世纪，捕捉图像的便捷性开启了解决多种问题的多样方法，无论是需要人工干预还是不需要。在本书中，我们涵盖了一些常见和现代任务，包括使用 PyTorch 进行图像分类、目标检测、图像分割、图像生成、操作生成的图像、利用基础模型完成各种任务、将计算机视觉与自然语言处理技术结合以及强化学习。我们从头开始学习了各种算法的工作细节。我们还了解了如何制定问题、捕获数据、训练模型并从训练模型中推断。我们理解了如何选择代码库/预训练模型并为我们的任务定制它们，最后，我们学习了如何以优化的方式部署我们的模型并加入漂移监测。

我们希望你已经掌握了处理图像的必要技能，就像它是你的第二天性一样，并且能够解决你感兴趣的任务。

最重要的是，我们希望这对你来说是一段愉快的旅程，希望你在阅读这本书时能和我们写作时一样享受！

# 问题

1.  什么是 REST API 以及它的作用是什么？

1.  Docker 是什么，为什么它对部署深度学习应用程序很重要？

1.  在生产环境中检测与训练时使用的图像明显不同的异常或新颖图像的一个简单而常见的技术是什么？

1.  如何加速大量向量图像的相似性搜索（百万级）？

# 在 Discord 上了解更多

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

[`packt.link/modcv`](https://packt.link/modcv)

![](img/QR_Code237402495622324343.png)
