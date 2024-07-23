# 第十二章：生产中深度学习端点监控

由于开发和生产设置的差异，一旦部署后**深度学习**（**DL**）模型的性能保证就变得困难。如果模型行为存在任何差异，必须在合理的时间内捕捉到；否则，会对下游应用产生负面影响。

在本章中，我们的目标是解释生产中监控 DL 模型行为的现有解决方案。我们将首先清楚地描述监控的好处以及保持整个系统稳定运行所需的条件。然后，我们将讨论监控 DL 模型和警报的流行工具。在介绍的各种工具中，我们将深入了解**CloudWatch**。我们将从 CloudWatch 的基础知识开始，讨论如何将 CloudWatch 集成到运行在**SageMaker**和**Elastic Kubernetes Service**（**EKS**）集群上的端点中。

在本章中，我们将覆盖以下主要话题：

+   生产中 DL 端点监控简介

+   使用 CloudWatch 进行监控

+   使用 CloudWatch 监控 SageMaker 端点

+   使用 CloudWatch 监控 EKS 端点

# 技术要求

您可以从本书的 GitHub 仓库下载本章的补充材料：[`github.com/PacktPublishing/Production-Ready-Applied-Deep-Learning/tree/main/Chapter_12`](https://github.com/PacktPublishing/Production-Ready-Applied-Deep-Learning/tree/main/Chapter_12)

# 生产中 DL 端点监控简介

我们将从描述部署端点的 DL 模型监控的好处开始本章。理想情况下，我们应该分析与传入数据、传出数据、模型指标和流量相关的信息。监控列出数据的系统可以为我们提供以下好处。

首先，*模型的输入和输出信息可以持久化存储在数据存储解决方案中（例如，Simple Storage Service（S3）存储桶），以便理解数据分布*。对传入数据和预测的详细分析有助于识别下游流程的潜在改进。例如，监控传入数据可以帮助我们识别模型预测中的偏差。在处理传入请求时，模型可能对特定特征组具有偏见。这些信息可以指导我们在为以下部署训练新模型时应考虑什么。另一个好处来自于模型的可解释性。出于业务目的或法律目的，需要解释模型预测的推理。这涉及到我们在*第九章*中描述的技术，*扩展深度学习管道*。

我们应该跟踪的另一个关键指标是端点的 **吞吐量**，这有助于我们提高用户满意度。*模型的行为可能会随着传入请求的数量和底层计算机的计算能力而变化*。我们可以监控推断延迟与传入流量之间的关系，以构建稳定高效的推断端点供用户使用。

在高层次上，DL 模型的监控可以分为两个领域：**端点监控** 和 **模型监控**。在前者领域，我们旨在收集与端点延迟和目标端点的吞吐量相关的数据。后者则专注于改善模型性能；我们需要收集传入数据、预测结果和模型性能，以及推断延迟。虽然许多模型监控用例通过在线方式在运行中的端点实现，但在训练和验证过程中也会以离线方式应用，目的是在部署前了解模型的行为。

在接下来的部分，我们将介绍用于监控 DL 模型的流行工具。

## 探索监控工具

监控工具主要可以分为两类，具体取决于它们的设计目标：**监控工具** 和 **警报工具**。详尽介绍所有工具超出了本书的范围；但我们将简要介绍其中的一些，以解释监控和警报工具的优势。请注意，界限通常不清晰，并且一些工具可能同时支持这两个功能。

让我们先来了解一下监控工具。

### Prometheus

**Prometheus** 是一个开源的监控和警报工具 ([`prometheus.io`](https://prometheus.io/))。Prometheus 将应用程序传递的数据存储在本地存储中。它使用时间序列数据库来存储、聚合和检索指标数据，这与监控任务的性质非常匹配。与 Prometheus 的交互涉及使用 **Prometheus 查询语言** (**PromQL**) ([`prometheus.io/docs/prometheus/latest/querying/basics`](https://prometheus.io/docs/prometheus/latest/querying/basics/))。Prometheus 设计用于处理诸如 **中央处理单元** (**CPU**) 使用情况、内存使用情况和延迟等指标。此外，还可以摄取用于监控的自定义指标，例如模型性能或传入和传出数据的分布。

### CloudWatch

**CloudWatch** 是由**亚马逊网络服务**（**AWS**）设计的监控和可观察性服务 ([`aws.amazon.com/cloudwatch`](https://aws.amazon.com/cloudwatch/))。与设置专用的 Prometheus 服务相比，CloudWatch 设置简单，因为它在幕后处理数据存储管理。默认情况下，大多数 AWS 服务如 AWS Lambda 和 EKS 集群使用 CloudWatch 持久化指标以供进一步分析。此外，CloudWatch 可以通过电子邮件或 Slack 消息通知用户关于受监视指标的异常变化。例如，您可以为指标设置阈值，并在其超过或低于预定义阈值时收到通知。有关警报功能的详细信息，请参阅 [`docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/AlarmThatSendsEmail.html`](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/AlarmThatSendsEmail.html)。

### Grafana

**Grafana** 是一个流行的工具，用于可视化从监控工具收集的指标 ([`grafana.com`](https://grafana.com/))。Grafana 可以读取来自 CloudWatch 或 AWS 管理的 Prometheus 的指标数据以进行可视化。关于这些配置的完整描述，建议您阅读 [`grafana.com/docs/grafana/latest/datasources/aws-cloudwatch`](https://grafana.com/docs/grafana/latest/datasources/aws-cloudwatch/) 和 [`docs.aws.amazon.com/prometheus/latest/userguide/AMP-onboard-query-standalone-grafana.html`](https://docs.aws.amazon.com/prometheus/latest/userguide/AMP-onboard-query-standalone-grafana.html)。

### Datadog

一种流行的专有解决方案是**Datadog** ([`www.datadoghq.com`](https://www.datadoghq.com/))。这个工具提供了多种监控功能：日志监控，应用性能监控，网络流量监控和实时用户监控。

### SageMaker Clarify

SageMaker 内置支持监控由 SageMaker 创建的端点，**SageMaker Clarify** ([`aws.amazon.com/sagemaker/clarify`](https://aws.amazon.com/sagemaker/clarify/))。SageMaker Clarify 带有一个**软件开发工具包**（**SDK**），有助于理解模型的性能及其在预测中的偏差。有关 SageMaker Clarify 的详细信息，请参阅 [`docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html`](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)。

在下一节中，我们将介绍警报工具。

## 探索警报工具

一个*事件*是需要后续操作的事件，比如失败的作业或构建。虽然监控工具可以捕获异常变化，但它们通常缺乏事件管理和响应过程的自动化。警报工具通过提供这些功能填补了这一空白。因此，许多公司通常集成明确的警报工具，以及时响应事件。

在本节中，我们将介绍两种最流行的警报工具：PagerDuty 和 Dynatrace。

### PagerDuty

作为警报和管理**事故响应**（**IR**）过程的工具，许多公司集成了**PagerDuty**（[PagerDuty](https://www.pagerduty.com/)）。在基本的警报功能之上，PagerDuty 支持根据事故类型和严重程度分配优先级。PagerDuty 可以从多个流行的监控软件中读取数据，比如[Prometheus 和 Datadog（https://aws.amazon.com/blogs/mt/using-amazon-managed-service-for-prometheus-alert-manager-to-re](https://aws.amazon.com/blogs/mt/using-amazon-managed-service-for-prometheus-alert-manager-to-receive-alerts-with-pagerduty/)ceive-alerts-with-pagerduty)。它还可以通过最小的代码更改与 CloudWatch 集成（[`support.pagerduty.com/docs/aws-cloudwatch-integration-guide`](https://support.pagerduty.com/docs/aws-cloudwatch-integration-guide)）。

### Dynatrace

**Dynatrace**是另一种专有工具，用于监控整个集群或网络和警报事件（[Dynatrace](https://www.dynatrace.com/)）。可以轻松监控正在运行的进程的资源使用情况、流量和响应时间。Dynatrace 具有基于警报配置文件的独特警报系统。这些配置文件定义了系统如何在整个组织中传递通知。Dynatrace 具有内置的推送通知功能，但也可以与其他提供通知功能的系统集成，例如 Slack 和 PagerDuty。

要记住的事情

a. 监控与端点相关的入站数据、出站数据、模型指标和流量量，使我们能够理解端点的行为，并帮助我们识别潜在的改进。

b. Prometheus 是一个开源的监控和警报系统，可用于监控 DL 端点的指标。CloudWatch 是 AWS 的监控服务，专为记录一组数据和跟踪入站和出站流量的异常变化而设计。

c. PagerDuty 是一种流行的警报工具，负责处理事故的完整生命周期。

在本节中，我们讨论了为什么需要对 DL 端点进行监控，并提供了可用工具的列表。在本章的其余部分，我们将详细研究 CloudWatch，这是最常见的监控工具，因为它与 AWS 中的大多数服务都很好地集成（例如 SageMaker）。

# 使用 CloudWatch 进行监控

首先，我们将介绍 CloudWatch 中的几个关键概念：日志、指标、报警和仪表板。**CloudWatch** 将摄入的数据以日志或按时间戳组织的指标形式持久化。如其名称所示，*日志* 指的是程序生命周期中发出的文本数据。另一方面，*指标* 表示组织的数值数据，如 CPU 或内存利用率。由于指标以有组织的方式存储，CloudWatch 支持从收集的数据中聚合指标并创建直方图。*报警* 可以设置为在目标指标报告异常变化时发出警报。此外，可以设置*仪表板*来直观地查看所选指标和已触发的报警。

在以下示例中，我们将描述如何使用 `boto3` 库中的 CloudWatch 服务客户端记录指标数据。指标数据结构化为字典，包含指标名称、维度和值。维度的概念是捕获有关指标的事实信息。例如，指标名称为 city 可以具有纽约市的值。然后，维度可以捕获特定信息，例如火灾或入室盗窃的每小时计数：

```py
import boto3
# create CloudWatch client using boto3 library
cloudwatch = boto3.client('cloudwatch')
# metrics data to ingest
data_metrics=[
    {
       'MetricName': 'gross_merchandise_value',
       'Dimensions': [
          {
             'Name': 'num_goods_sold',
             'Value': '369'
          } ],
       'Unit': 'None',
       'Value': 900000.0
    } ]
# ingest the data for monitoring 
cloudwatch.put_metric_data(
    MetricData=data_metrics, # data for metrics 
    Namespace='ECOMMERCE/Revenue' # namespace to separate domain/projects)
```

在前面的代码片段中，我们首先使用 `boto3.client` 函数为 CloudWatch 创建一个 `cloudwatch` 服务客户端实例。这个实例将允许我们从 Python 环境与 CloudWatch 进行通信。记录数据的关键方法是 `put_metric_data`。这个函数的 `put_metric_data` 方法从 CloudWatch 客户端实例接收 `MetricData`（要摄入到 CloudWatch 中的目标指标数据：`data_metrics`）和 `Namespace`（容器，用于容纳指标数据：`ECOMMERCE/Revenue`）。不同命名空间的数据被单独管理，以支持高效的聚合。

在这个示例中，`data_metrics` 指标数据包含一个名为 `MetricName` 的字段，其值为 `gross_merchandise_value`，值为 `900000.0`。`gross_merchandise_value` 的单位被定义为 `None`。此外，我们还提供了销售商品数 (`num_goods_sold`) 作为额外的维度信息。

要获取完整的 CloudWatch 概念说明，请参阅 [`docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/cloudwatch_concepts.html`](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/cloudwatch_concepts.html)。

需要记住的事情

a. CloudWatch 将摄入的数据以日志或按时间戳组织的指标形式持久化。它支持为异常变化设置报警，并通过仪表板提供有效的可视化。

b. 使用 `boto3` 库可以轻松地将一个指标记录到 CloudWatch。它提供了一个支持通过 `put_metric_data` 函数进行日志记录的 CloudWatch 服务客户端。

尽管云观察的日志可以像本节描述的那样明确地记录，但 SageMaker 为一些开箱即用的指标提供了内置的日志记录功能。让我们仔细看看它们。

# 使用 CloudWatch 监视 SageMaker 端点

作为**机器学习**的端到端服务，SageMaker 是我们实施 DL 项目各个步骤的主要工具之一。在本节中，我们将描述最后一个缺失的部分：监控使用 SageMaker 创建的端点。首先，我们将解释如何设置基于 CloudWatch 的训练监控，其中指标以离线批次报告。接下来，我们将讨论如何监控实时端点。

本节中的代码片段设计用于在 SageMaker Studio 上运行。因此，我们首先需要定义 AWS **Identity and Access Management** (**IAM**) 角色和一个会话对象。让我们看看第一个代码片段：

```py
import sagemaker
# IAM role of the notebook
role_exec=sagemaker.get_execution_role()
# a sagemaker session object
sag_sess=sagemaker.session()
```

在前面的代码片段中，`get_execution_role` 函数提供笔记本的 IAM 角色。`role_exec`。`sagemaker.session` 提供了 SageMaker `sag_sess` SageMaker 会话对象，用于作业配置。

## 在 SageMaker 中监控模型的整个训练过程

在模型训练期间的日志记录涉及 SageMaker 的 `Estimator` 类。它可以使用 `regex` 表达式处理打印的消息，并将它们存储为指标。您可以在此处看到一个例子：

```py
import sagemaker
from sagemaker.estimator import Estimator
# regex pattern for capturing error metrics 
reg_pattern_metrics=[
   {'Name':'train:error','Regex':'Train_error=(.*?);'},
   {'Name':'validation:error','Regex':'Valid_error=(.*?)'}]
# Estimator instance for model training
estimator = Estimator(
   image_uri=...,
   role=role_exec,
   sagemaker_session=sag_sess,
   instance_count=...,
   instance_type=...,
   metric_definitions=reg_pattern_metrics)
```

在前面的代码片段中，我们创建了 `estimator`，这是用于训练的 `Estimator` 实例。大多数参数的解释可以在 *第六章*，*高效模型训练* 中找到。在此示例中，我们正在定义的附加参数是 `metric_definitions`。我们传递的是 `reg_pattern_metrics`，它定义了一组 `Train_error=(.*?)` 和 `Valid_error=(.*?)`，训练和评估日志。匹配给定模式的文本将作为指标持久保存在 CloudWatch 中。有关使用 `Estimator` 类在整个模型训练过程中进行离线指标记录的完整详情，请参阅 [`docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html`](https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html)。我们想要提到的是，特定的训练作业指标（如内存、CPU、**图形处理单元** (**GPU**) 和磁盘利用率）会自动记录，并且您可以通过 CloudWatch 或 SageMaker 控制台监视它们。

## 监控来自 SageMaker 的实时推断端点

在本节中，我们将描述 SageMaker 的端点基于 CloudWatch 的监控功能。在下面的代码片段中，我们呈现了一个带有 `output_handler` 函数的样本 `inference.py` 脚本。该文件被分配为 SageMaker 的 `Model` 或 `Estimator` 类的 `entry_point` 参数，以定义额外的预处理和后处理逻辑。有关 `inference.py` 的详细信息，请参阅 *第九章*，*扩展深度学习管道*。`output_handler` 函数设计用于处理模型预测并使用 `print` 函数记录度量数据。打印的消息作为日志存储在 CloudWatch 中：

```py
# inference.py
def output_handler(data, context):
    # retrieve the predictions
    results=data.content
    # data that will be ingested to CloudWatch
    data_metrics=[
       {
          'MetricName': 'model_name',
          'Dimensions': [
             {
                'Name': 'classify',
                'Value': results
              } ],
          'Unit': 'None',
          'Value': "classify_applicant_risk"
      } ]
    # print will ingest information into CloudWatch
    print(data_metrics)
```

在前面的推断代码中，我们首先获得模型预测（`results`），并为度量数据构建一个字典（`data_metrics`）。该字典已经具有`MetricName` 值为 `model_name` 和名为 `classify` 的维度。模型预测将被指定为 `classify` 维度。SageMaker 将收集打印的度量数据并将其输入到 CloudWatch。有关在这种场景下连续模型质量漂移监控的示例方法，请参阅在线说明 [`sagemaker-examples.readthedocs.io/en/latest/sagemaker_model`](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker_model_monitor/model_quality/model_quality_churn_sdk.html)_monitor/model_quality/model_quality_churn_sdk.html。这页详细解释了如何在这些情况下利用 CloudWatch。

需记住的事项

a. SageMaker 的 `Estimator` 类在训练期间提供了对基于 CloudWatch 的监控的内置支持。在构造实例时，您需要将一组正则表达式模式传递给 `metric_definitions` 参数。

b. SageMaker 端点的打印消息将存储为 CloudWatch 日志。因此，我们可以通过记录度量数据的 `entry_point` 脚本来实现监控。

在本节中，我们解释了 SageMaker 如何支持基于 CloudWatch 的监控。让我们看看 EKS 如何支持推断端点的监控。

# 使用 CloudWatch 监控 EKS 端点

除了 SageMaker 外，我们还在《*第九章*》中描述了基于 EKS 的端点，在*扩展深度学习流水线*中。在本节中，我们描述了 EKS 可用的基于 CloudWatch 的监控。首先，我们将学习如何从容器中记录 EKS 指标以进行监控。接下来，我们将解释如何从 EKS 推断端点记录与模型相关的度量数据。

让我们首先看看如何设置 CloudWatch 来监控 EKS 集群。最简单的方法是在容器中安装 CloudWatch 代理。此外，您还可以安装 **Fluent Bit**，这是一个开源工具，进一步增强了日志记录过程（[www.fluentbit.io](http://www.fluentbit.io)）。有关 CloudWatch 代理和 Fluent Bit 的完整说明，请阅读 [`docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Container-Insights-setup-EKS-quickstart.html`](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Container-Insights-setup-EKS-quickstart.html)。

另一个选项是保留由 EKS 控制平面发送的默认指标。这可以通过 EKS Web 控制台轻松启用（[`docs.aws.amazon.com/eks/latest/userguide/control-plane-logs.html`](https://docs.aws.amazon.com/eks/latest/userguide/control-plane-logs.html)）。可以在[`aws.github.io/aws-eks-best-practices/reliability/docs/controlplane`](https://aws.github.io/aws-eks-best-practices/reliability/docs/controlplane/)找到 EKS 控制平面发出的完整指标列表。例如，如果您对记录与延迟相关的指标感兴趣，可以使用`apiserver_request_duration_seconds*`。

在模型推断期间记录模型相关的指标，您需要在代码中实例化`boto3`的 CloudWatch 服务客户端，并显式记录它们。前一节中包含的代码片段，*使用 CloudWatch 监视 SageMaker 端点*，应该是一个很好的起点。

记住的事情

a. 从一个 EKS 集群记录端点相关的指标可以通过使用 CloudWatch 代理或保留由 EKS 控制平面发送的默认指标来实现。

b. 使用`boto3`库需要显式记录与模型相关的指标。

作为本节的最后一个主题，我们解释了如何从 EKS 集群将各种指标记录到 CloudWatch。

# 摘要

本章的目标是解释为什么需要监视运行 DL 模型的端点，并介绍该领域中的流行工具。我们在本章介绍的工具旨在监控端点的信息集并在监控指标发生突变时提供警报。我们涵盖的工具包括 CloudWatch、Prometheus、Grafana、Datadog、SageMaker Clarify、PagerDuty 和 Dynatrace。为了完整起见，我们还介绍了如何将 CloudWatch 集成到 SageMaker 和 EKS 中，以监视端点及模型性能。

在下一章中，作为本书的最后一章，我们将探讨评估已完成项目的过程并讨论潜在的改进。
