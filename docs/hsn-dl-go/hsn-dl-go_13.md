# 扩展部署

现在我们已经介绍了一个管理数据流水线的工具，现在是完全深入了解的时候了。我们的模型最终在软件的多层抽象下运行在我们在第五章中讨论过的硬件上，直到我们可以使用像`go build --tags=cuda`这样的代码为止。

我们在 Pachyderm 上构建的图像识别流水线部署是本地的。我们以一种与部署到云资源相同的方式进行了部署，而不深入探讨其具体外观。现在我们将专注于这一细节。

通过本章末尾，您应能够做到以下几点：

+   识别和理解云资源，包括我们平台示例中的特定资源（AWS）。

+   知道如何将您的本地部署迁移到云上

+   理解 Docker 和 Kubernetes 以及它们的工作原理。

+   理解计算与成本之间的权衡。

# 在云中迷失（和找到）

拥有一台配备 GPU 和 Ubuntu 系统的强大台式机非常适合原型设计和研究，但是当你需要将模型投入生产并实际进行每日预测时，你需要高可用性和可扩展性的计算资源。这究竟意味着什么？

想象一下，你已经拿我们的**卷积神经网络**（**CNN**）例子，调整了模型并用自己的数据进行了训练，并创建了一个简单的 REST API 前端来调用模型。你想要围绕提供客户服务的一个小业务，客户支付一些费用，获得一个 API 密钥，可以提交图像到一个端点并获得一个回复，说明图像包含什么内容。作为服务的图像识别！听起来不错吧？

我们如何确保我们的服务始终可用和快速？毕竟，人们付费给你，即使是小的停机或可靠性下降也可能导致您失去客户。传统上的解决方案是购买一堆昂贵的*服务器级*硬件，通常是带有多个电源和网络接口的机架式服务器，以确保在硬件故障的情况下服务的连续性。您需要检查每个级别的冗余选项，从磁盘或存储到网络，甚至到互联网连接。

据说你需要两个一模一样的东西，这一切都需要巨大甚至是禁止性的成本。如果你是一家大型、有资金支持的初创公司，你有很多选择，但当然，随着资金曲线的下降，你的选择也会减少。自助托管变成了托管托管（并非总是如此，但对于大多数小型或初创用例而言是如此），这反过来又变成了存储在别人数据中心中的计算的标准化层，以至于你根本不需要关心底层硬件或基础设施。

当然，在现实中，并非总是如此。像 AWS 这样的云提供商将大部分乏味、痛苦（但必要）的工作，如硬件更换和常规维护，从中排除了。你不会丢失硬盘或陷入故障的网络电缆，如果你决定（*嘿，一切都运作良好*），为每天 10 万客户提供服务，那么你只需推送一个简单的基础架构规格变更。不需要给托管提供商打电话，协商停机时间，或去计算机硬件店。

这是一个非常强大的想法；你解决方案的实际执行——硅和装置的混合物，用于做出预测——几乎可以被视为一种事后的考虑，至少与几年前相比是如此。一般来说，维护云基础设施所需的技能集或方法被称为**DevOps**。这意味着一个人同时参与两个（或更多！）阵营。他们了解这些 AWS 资源代表什么（服务器、交换机和负载均衡器），以及如何编写必要的代码来指定和管理它们。

一个不断发展的角色是*机器学习工程师*。这是传统的 DevOps 技能集，但随着更多的*Ops*方面被自动化或抽象化，个人也可以专注于模型训练或部署，甚至扩展。让工程师参与整个堆栈是有益的。理解一个模型可并行化的程度，一个特定模型可能具有的内存需求，以及如何构建必要的分布式基础设施以进行大规模推理，所有这些都导致了一个模型服务基础设施，在这里各种设计元素不是领域专业化的产物，而是一个整体的集成。

# 构建部署模板

现在我们将组合所需的各种模板，以便在规模上部署和训练我们的模型。这些模板包括：

+   **AWS 云形成模板**：虚拟实例及相关资源

+   **Kubernetes 或 KOPS 配置**：K8s 集群管理

+   **Docker 模板或 Makefile**：创建图像以部署在我们的 K8s 集群上

我们在这里选择了一条特定的路径。AWS 有诸如**弹性容器服务**（**ECS**）和**弹性 Kubernetes 服务**（**EKS**）等服务，通过简单的 API 调用即可访问。我们在这里的目的是深入了解细节，以便您可以明智地选择如何扩展部署您自己的用例。目前，您可以更精细地控制容器选项，以及在将容器部署到纯净的 EC2 实例时如何调用您的模型。这些服务在成本和性能方面的权衡将在稍后的成本和性能折衷部分中进行讨论。

# 高级步骤

我们的迷你 CI/CD 流水线包括以下任务：

1.  创建或推送训练或推理 Docker 镜像到 AWS ECS。

1.  在 EC2 实例上创建或部署一个带有 Kubernetes 集群的 AWS 堆栈，以便我们可以执行下一步操作。

1.  训练一个模型或进行一些预测！

现在我们将依次详细介绍每个步骤的细节。

# 创建或推送 Docker 镜像

Docker 肯定是一个引起了很多炒作的工具。除了人类时尚之外，其主要原因在于 Docker 简化了诸如依赖管理和模型集成等问题，使得可重复使用、广泛部署的构建成为可能。我们可以预先定义从操作系统获取的所需内容，并在我们知道依赖项是最新的时候将它们全部打包起来，这样我们所有的调整和故障排除工作都不会徒劳无功。

要创建我们的镜像并将其送到我们想要去的地方，我们需要两样东西：

+   **Dockerfile**：这定义了我们的镜像、Linux 的版本、要运行的命令以及在启动容器时运行的默认命令。

+   **Makefile**：这将创建镜像并将其推送到 AWS ECS。

让我们先看一下 Dockerfile：

```py
FROM ubuntu:16.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
 curl \
 git \
 pkg-config \
 rsync \
 awscli \
 wget \
 && \
 apt-get clean && \
 rm -rf /var/lib/apt/lists/*

RUN wget -nv https://storage.googleapis.com/golang/go1.12.1.linux-amd64.tar.gz && \
 tar -C /usr/local -xzf go1.12.1.linux-amd64.tar.gz

ENV GOPATH /home/ubuntu/go

ENV GOROOT /usr/local/go

ENV PATH $PATH:$GOROOT/bin

RUN /usr/local/go/bin/go version && \
 echo $GOPATH && \
 echo $GOROOT

RUN git clone https://github.com/PacktPublishing/Hands-On-Deep-Learning-with-Go

RUN go get -v gorgonia.org/gorgonia && \
 go get -v gorgonia.org/tensor && \
 go get -v gorgonia.org/dawson && \
 go get -v github.com/gogo/protobuf/gogoproto && \
 go get -v github.com/golang/protobuf/proto && \
 go get -v github.com/google/flatbuffers/go && \
 go get -v .

WORKDIR /

ADD staging/ /app

WORKDIR /app

CMD ["/bin/sh", "model_wrapper.sh"]
```

我们可以通过查看每行开头的大写声明来推断一般方法：

1.  选择基础 OS 镜像使用`FROM`。

1.  使用`ARG`设置启动。

1.  使用`RUN`运行一系列命令，使我们的 Docker 镜像达到期望的状态。然后将一个`staging`数据目录添加到`/app`。

1.  切换到新的`WORKDIR`。

1.  执行`CMD`命令，我们的容器将运行。

现在我们需要一个 Makefile。这个文件包含了将构建我们刚刚在 Dockerfile 中定义的镜像并将其推送到亚马逊容器托管服务 ECS 的命令。

这是我们的 Makefile：

```py
cpu-image:
 mkdir -p staging/
 cp model_wrapper.sh staging/
 docker build --no-cache -t "ACCOUNTID.dkr.ecr.ap-southeast-2.amazonaws.com/$(MODEL_CONTAINER):$(VERSION_TAG)" .
 rm -rf staging/

cpu-push: cpu-image
 docker push "ACCOUNTID.dkr.ecr.ap-southeast-2.amazonaws.com/$(MODEL_CONTAINER):$(VERSION_TAG)"

```

与我们已经涵盖过的其他示例一样，我们正在使用`sp-southeast-2`地区；但是，您可以自由指定您自己的地区。您还需要包含您自己的 12 位 AWS 帐户 ID。

从这个目录（时间未到时，请耐心等待！）我们现在可以创建和推送 Docker 镜像。

# 准备您的 AWS 账户

您将看到有关 API 访问 AWS 的通知，以便 KOPS 管理您的 EC2 和相关计算资源。与此 API 密钥相关联的帐户还需要以下 IAM 权限：

+   AmazonEC2FullAccess

+   AmazonRoute53FullAccess

+   AmazonS3FullAccess

+   AmazonVPCFullAccess

您可以通过进入 AWS 控制台并按照以下步骤操作来启用程序化或 API 访问：

1.  点击 IAM

1.  从左侧菜单中选择“用户”，然后选择您的用户

1.  选择“安全凭证”。然后，您将看到“访问密钥”部分

1.  点击“创建访问密钥”，然后按照说明操作

结果产生的密钥和密钥 ID 将被用于您的 `~/.aws/credentials` 文件或作为 shell 变量导出，以供 KOPS 和相关部署和集群管理工具使用。

# 创建或部署 Kubernetes 集群

我们的 Docker 镜像必须运行在某个地方，为什么不是一组 Kubernetes pod 呢？这正是分布式云计算的魔力所在。使用中央数据源，例如 AWS S3，在我们的情况下，会为训练或推理启动许多微实例，最大化 AWS 资源利用率，为您节省资金，并为企业级机器学习应用提供所需的稳定性和性能。

首先，进入存放这些章节的仓库中的 `/k8s/` 目录。

我们将开始创建部署集群所需的模板。在我们的案例中，我们将使用 `kubectl` 的前端，默认的 Kubernetes 命令，它与主 API 进行交互。

# Kubernetes

让我们来看看我们的 `k8s_cluster.yaml` 文件：

```py
apiVersion: kops/v1alpha2
kind: Cluster
metadata:
  creationTimestamp: 2018-05-01T12:11:24Z
  name: $NAME
spec:
  api:
    loadBalancer:
      type: Public
  authorization:
    rbac: {}
  channel: stable
  cloudProvider: aws
  configBase: $KOPS_STATE_STORE/$NAME
  etcdClusters:
  - etcdMembers:
    - instanceGroup: master-$ZONE
      name: b
    name: main
  - etcdMembers:
    - instanceGroup: master-$ZONE
      name: b
    name: events
  iam:
    allowContainerRegistry: true
    legacy: false
  kubernetesApiAccess:
  - 0.0.0.0/0
  kubernetesVersion: 1.9.3
  masterInternalName: api.internal.$NAME
  masterPublicName: api.hodlgo.$NAME
  networkCIDR: 172.20.0.0/16
  networking:
    kubenet: {}
  nonMasqueradeCIDR: 100.64.0.0/10
  sshAccess:
  - 0.0.0.0/0
  subnets:
  - cidr: 172.20.32.0/19
    name: $ZONE
    type: Public
    zone: $ZONE
  topology:
    dns:
      type: Public
    masters: public
    nodes: public
```

让我们来看看我们的 `k8s_master.yaml` 文件：

```py
apiVersion: kops/v1alpha2
kind: InstanceGroup
metadata:
  creationTimestamp: 2018-05-01T12:11:25Z
  labels:
    kops.k8s.io/cluster: $NAME
  name: master-$ZONE
spec:
  image: kope.io/k8s-1.8-debian-jessie-amd64-hvm-ebs-2018-02-08
  machineType: $MASTERTYPE
  maxSize: 1
  minSize: 1
  nodeLabels:
    kops.k8s.io/instancegroup: master-$ZONE
  role: Master
  subnets:
  - $ZONE
```

让我们来看看我们的 `k8s_nodes.yaml` 文件：

```py
apiVersion: kops/v1alpha2
kind: InstanceGroup
metadata:
  creationTimestamp: 2018-05-01T12:11:25Z
  labels:
    kops.k8s.io/cluster: $NAME
  name: nodes-$ZONE
spec:
  image: kope.io/k8s-1.8-debian-jessie-amd64-hvm-ebs-2018-02-08
  machineType: $SLAVETYPE
  maxSize: $SLAVES
  minSize: $SLAVES
  nodeLabels:
    kops.k8s.io/instancegroup: nodes-$ZONE
  role: Node
  subnets:
  - $ZONE
```

这些模板将被输入到 Kubernetes 中，以便启动我们的集群。我们将用来部署集群和相关 AWS 资源的工具是 *KOPS*。在撰写本文时，该工具的当前版本为 1.12.1，并且所有部署均已使用此版本进行测试；较早版本可能存在兼容性问题。

首先，我们需要安装 KOPS。与我们之前的所有示例一样，这些步骤也适用于 macOS。我们使用 Homebrew 工具来管理依赖项，并保持安装的局部化和合理性：

```py
#brew install kops
==> Installing dependencies for kops: kubernetes-cli
==> Installing kops dependency: kubernetes-cli
==> Downloading https://homebrew.bintray.com/bottles/kubernetes-cli-1.14.2.mojave.bottle.tar.gz
==> Downloading from https://akamai.bintray.com/85/858eadf77396e1acd13ddcd2dd0309a5eb0b51d15da275b491
######################################################################## 100.0%
==> Pouring kubernetes-cli-1.14.2.mojave.bottle.tar.gz
==> Installing kops
==> Downloading https://homebrew.bintray.com/bottles/kops-1.12.1.mojave.bottle.tar.gz
==> Downloading from https://akamai.bintray.com/86/862c5f6648646840c75172e2f9f701cb590b04df03c38716b5
######################################################################## 100.0%
==> Pouring kops-1.12.1.mojave.bottle.tar.gz
==> Caveats
Bash completion has been installed to:
 /usr/local/etc/bash_completion.d

zsh completions have been installed to:
 /usr/local/share/zsh/site-functions
==> Summary
 /usr/local/Cellar/kops/1.12.1: 5 files, 139.2MB
==> Caveats
==> kubernetes-cli
Bash completion has been installed to:
 /usr/local/etc/bash_completion.d

zsh completions have been installed to:
 /usr/local/share/zsh/site-functions
==> kops
Bash completion has been installed to:
 /usr/local/etc/bash_completion.d

zsh completions have been installed to:
 /usr/local/share/zsh/site-functions

```

我们可以看到 KOPS 已经安装，连同 `kubectl` 一起，这是与 API 直接交互的默认 K8s 集群管理工具。请注意，Homebrew 经常会输出关于命令完成的警告类型消息，可以安全地忽略这些消息；但是，如果出现关于符号链接配置的错误，请按照说明解决与任何现有的 `kubectl` 本地安装冲突的问题。

# 集群管理脚本

我们还需要编写一些脚本，以允许我们设置环境变量并根据需要启动或关闭 Kubernetes 集群。在这里，我们将整合我们编写的模板，KOPS 或 `kubectl`，以及我们在前几节中完成的 AWS 配置。

让我们来看看我们的 `vars.sh` 文件：

```py
#!/bin/bash

# AWS vars
export BUCKET_NAME="hodlgo-models"
export MASTERTYPE="m3.medium"
export SLAVETYPE="t2.medium"
export SLAVES="2"
export ZONE="ap-southeast-2b"

# K8s vars
export NAME="hodlgo.k8s.local"
export KOPS_STATE_STORE="s3://hodlgo-cluster"
export PROJECT="hodlgo"
export CLUSTER_NAME=$PROJECT

# Docker vars
export VERSION_TAG="0.1"
export MODEL_CONTAINER="hodlgo-model"
```

我们可以看到这里的主要变量是容器名称、K8s 集群详细信息以及我们想要启动的 AWS 资源种类的一堆规格（及其所在的区域）。您需要用您自己的值替换这些值。

现在，我们可以编写相应的脚本，在完成部署或管理 K8s 集群后，清理我们的 shell 中的变量是一个重要部分。

让我们看看我们的 `unsetvars.sh` 文件：

```py
#!/bin/bash

# Unset them vars

unset BUCKET_NAME
unset MASTERTYPE
unset SLAVETYPE
unset SLAVES
unset ZONE

unset NAME
unset KOPS_STATE_STORE

unset PROJECT
unset CLUSTER_NAME

unset VERSION_TAG
unset MODEL_CONTAINER
```

现在，启动我们的集群脚本将使用这些变量来确定集群的命名方式、节点数量以及部署位置。您将看到我们在一行中使用了一个小技巧来将环境变量传递到我们的 Kubernetes 模板或 KOPS 中；在未来的版本中，这可能不再需要，但目前这是一个可行的解决方法。

让我们看看我们的 `cluster-up.sh` 文件：

```py
#!/bin/bash

## Bring up the cluster with kops

set -e

echo "Bringing up Kubernetes cluster"
echo "Using Cluster Name: ${CLUSTER_NAME}"
echo "Number of Nodes: ${SLAVES}"
echo "Using Zone: ${ZONE}"
echo "Bucket name: ${BUCKET_NAME}"

export PARALLELISM="$((4 * ${SLAVES}))"

# Includes ugly workaround because kops is unable to take stdin as input to create -f, unlike kubectl
cat k8s_cluster.yaml | envsubst > k8s_cluster-edit.yaml && kops create -f k8s_cluster-edit.yaml
cat k8s_master.yaml | envsubst > k8s_master-edit.yaml && kops create -f k8s_master-edit.yaml
cat k8s_nodes.yaml | envsubst > k8s_nodes-edit.yaml && kops create -f k8s_nodes-edit.yaml

kops create secret --name $NAME sshpublickey admin -i ~/.ssh/id_rsa.pub
kops update cluster $NAME --yes

echo ""
echo "Cluster $NAME created!"
echo ""

# Cleanup from workaround
rm k8s_cluster-edit.yaml
rm k8s_master-edit.yaml
rm k8s_nodes-edit.yaml
```

相应的 `down` 脚本将关闭我们的集群，并确保相应清理 AWS 资源。

让我们看看我们的 `cluster-down.sh` 文件：

```py
#!/bin/bash

## Kill the cluster with kops

set -e

echo "Deleting cluster $NAME"
kops delete cluster $NAME --yes
```

# 构建和推送 Docker 容器

现在我们已经做好了准备，准备好了所有模板和脚本，我们可以继续实际制作 Docker 镜像，并将其推送到 ECR，以便进行完整集群部署之前使用。

首先，我们导出了本章前面生成的 AWS 凭证：

```py
export AWS_DEFAULT_REGION=ap-southeast-2
export AWS_ACCESS_KEY_ID="<your key here>"
export AWS_SECRET_ACCESS_KEY="<your secret here>"
```

然后，我们获取容器存储库登录信息。这是必要的，以便我们可以推送创建的 Docker 镜像到 ECR，然后由我们的 Kubernetes 节点在模型训练或推理时拉取。请注意，此步骤假定您已安装 AWS CLI：

```py
aws ecr get-login --no-include-email
```

此命令的输出应类似于以下内容：

```py
docker login -u AWS -p xxxxx https://ACCOUNTID.dkr.ecr.ap-southeast-2.amazonaws.com
```

然后，我们可以执行 `make cifarcnn-image` 和 `make cifarcnn-push`。这将构建 Dockerfile 中指定的 Docker 镜像，并将其推送到 AWS 的容器存储服务中。

# 在 K8s 集群上运行模型

您现在可以编辑我们之前创建的 `vars.sh` 文件，并使用您喜爱的命令行文本编辑器设置适当的值。您还需要创建一个存储 k8s 集群信息的存储桶。

完成这些步骤后，您可以启动您的 Kubernetes 集群：

```py
source vars.sh
./cluster-up.sh
```

现在，KOPS 正通过 `kubectl` 与 Kubernetes 进行交互，以启动将运行您的集群的 AWS 资源，并在这些资源上配置 K8s 本身。在继续之前，您需要验证您的集群是否已成功启动：

```py
kops validate cluster
Validating cluster hodlgo.k8s.local

INSTANCE GROUPS
NAME ROLE MACHINETYPE MIN MAX SUBNETS
master-ap-southeast-2a Master c4.large 1 1 ap-southeast-2
nodes Node t2.medium 2 2 ap-southeast-2

NODE STATUS
NAME ROLE READY
ip-172-20-35-114.ec2.internal node True
ip-172-20-49-22.ec2.internal master True
ip-172-20-64-133.ec2.internal node True
```

一旦所有 K8s 主节点返回 `Ready`，您就可以开始在整个集群节点上部署您的模型！

执行此操作的脚本很简单，并调用 `kubectl` 以与我们的 `cluster_up.sh` 脚本相同的方式应用模板。

让我们看看我们的 `deploy-model.sh` 文件：

```py
#!/bin/bash

# envsubst doesn't exist for OSX. needs to be brew-installed
# via gettext. Should probably warn the user about that.
command -v envsubst >/dev/null 2>&1 || {
  echo >&2 "envsubst is required and not found. Aborting"
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo >&2 "------------------------------------------------"
    echo >&2 "If you're on OSX, you can install with brew via:"
    echo >&2 " brew install gettext"
    echo >&2 " brew link --force gettext"
  fi
  exit 1;
}

cat ${SCRIPT_DIR}/model.yaml | envsubst | kubectl apply -f -

```

# 概要

现在，我们来详细介绍 Kubernetes、Docker 和 AWS 的底层细节，以及如何根据您的钱包能力将尽可能多的资源投入到模型中。接下来，您可以采取一些步骤，定制这些示例以适应您的用例，或者进一步提升您的知识水平：

+   将这种方法集成到您的 CI 或 CD 工具中（如 Bamboo、CircleCI、Puppet 等）

+   将 Pachyderm 集成到您的 Docker、Kubernetes 或 AWS 解决方案中

+   使用参数服务器进行实验，例如分布式梯度下降，进一步优化您的模型流水线
