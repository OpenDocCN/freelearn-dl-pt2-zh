# 7 个用于数据科学的 LLMs

## 在 Discord 上加入我们的书籍社区

[`packt.link/EarlyAccessCommunity`](https://packt.link/EarlyAccessCommunity)

![自动生成的 Qr 代码说明](img/file49.png)

这一章介绍了生成式人工智能如何自动化数据科学。生成式人工智能，特别是大型语言模型（LLMs），有望加速各种领域的科学进展，尤其是通过提供对研究数据的高效分析和帮助文献综述过程。许多现有的自动机器学习（AutoML）方法可以帮助数据科学家提高生产力，并帮助使数据科学更具可重复性。我将先概述数据科学中的自动化，然后我们将讨论生成式人工智能如何影响数据科学。接下来，我们将讨论如何使用代码生成和工具以不同方式回答与数据科学相关的问题。这可以采用模拟的形式或通过附加信息来丰富数据集。最后，我们将关注结构化数据集的探索性分析。我们可以设置代理人在 Pandas 中运行 SQL 或表格数据。我们将看到如何对数据集提出问题、有关数据的统计问题或要求可视化。在整个章节中，我们将使用 LLMs 的不同方法进行数据科学，您可以在本书的 Github 存储库中的`data_science`目录中找到。[`github.com/benman1/generative_ai_with_langchain`](https://github.com/benman1/generative_ai_with_langchain)。主要部分包括：

+   自动化数据科学

+   代理人可以回答数据科学问题

+   LLMs 的数据探索

首先，让我们讨论数据科学如何自动化，以及自动化的哪些部分，以及生成式人工智能将如何影响数据科学。

## 自动化数据科学

数据科学是将计算机科学、统计学和商业分析相结合，从数据中提取知识和洞见的领域。数据科学家使用各种工具和技术来收集、清理、分析和可视化数据。然后，他们利用这些信息帮助企业做出更好的决策。数据科学家的工作可能因具体职位和行业而有所不同。但是，数据科学家可能执行的一些常见任务包括：

+   收集数据：数据科学家需要从各种来源收集数据，例如数据库、社交媒体和传感器。

+   清理数据：数据科学家需要清理数据以删除错误和不一致性。

+   分析数据：数据科学家使用各种统计和机器学习技术来分析数据。

+   可视化数据：数据科学家使用数据可视化来向利益相关者传达见解。

+   建立模型：数据科学家建立模型来预测未来结果或做出建议。

数据分析是数据科学的一个子集，专注于从数据中提取洞察。数据分析师使用各种工具和技术来分析数据，但是他们通常不构建模型。数据科学和数据分析的重叠在于两个领域都涉及使用数据提取洞察。然而，与数据分析师相比，数据科学家通常具有更高技术水平的技能。数据科学家也更有可能构建模型，有时将模型部署到生产环境中。数据科学家有时会将模型部署到生产环境中，以便实时使用它们来做决策。但是，在本文中，我们将避免自动部署模型。下面是总结数据科学和数据分析之间关键差异的表格：

| 特征 | 数据科学 | 数据分析 |
| --- | --- | --- |
| 技术技能 | 更高 | 较低 |
| 机器学习 | 是 | 否 |
| 模型部署 | 有时 | 否 |
| 重点 | 提取洞察和建模 | 提取洞察 |

图 7.1：数据科学和数据分析的比较。

两者之间的共同点是收集数据、清理数据、分析数据、可视化数据，所有这些都属于提取洞见的范畴。此外，数据科学还涉及训练机器学习模型，通常更加注重统计学。在某些情况下，根据公司的设置和行业惯例，部署模型和编写软件可能会被添加到数据科学的任务清单中。自动化数据分析和数据科学旨在自动化与数据处理相关的许多乏味重复的任务。这包括数据清理、特征工程、模型训练、调优和部署。目标是通过实现更快的迭代和减少常见工作流程的手动编码来提高数据科学家和分析师的生产力。许多这些任务可以在一定程度上自动化。数据科学的一些任务与我们在第六章《软件开发》中谈到的软件开发者的任务类似，即编写和部署软件，尽管焦点更窄，专注于模型。数据科学平台（如 Weka、H2O、KNIME、RapidMiner 和 Alteryx）是统一的机器学习和分析引擎，可用于各种任务，包括大容量数据的预处理和特征提取。所有这些都配有图形用户界面（GUI），具有集成第三方数据源和编写自定义插件的能力。KNIME 主要是开源的，但也提供了一个名为 KNIME Server 的商业产品。Apache Spark 是一种多功能工具，可用于数据科学中涉及的各种任务。它可用于清理、转换、提取特征和准备高容量数据进行分析，还可用于训练和部署机器学习模型，无论是在流式处理方案中，还是在实时决策或监视事件时。此外，在其最基本的层面上，科学计算库（如 NumPy）可以用于自动化数据科学中涉及的所有任务。深度学习和机器学习库（如 TensorFlow、Pytorch 和 Scikit-Learn）可以用于各种任务，包括数据预处理和特征提取。编排工具（如 Airflow、Kedro 或其他工具）可以帮助完成所有这些任务，并与数据科学的各个步骤相关的特定工具进行了大量集成。几种数据科学工具都支持生成式 AI。在《第六章》《软件开发》中，我们已经提到了 GitHub Copilot，但还有其他工具，如 PyCharm AI 助手，甚至更加具体的 Jupyter AI，这是 Project Jupyter 的一个子项目，将生成式人工智能引入到 Jupyter 笔记本中。Jupyter AI 允许用户生成代码、修复错误、总结内容，甚至使用自然语言提示创建整个笔记本。该工具将 Jupyter 与来自各种提供者的 LLM 连接起来，使用户可以选择其首选模型和嵌入。Jupyter AI 优先考虑负责任的人工智能和数据隐私。底层提示、链和组件是开源的，确保透明度。它保存有关模型生成内容的元数据，使得在工作流程中跟踪 AI 生成的代码变得容易。Jupyter AI 尊重用户数据隐私，只有在明确请求时才会与 LLM 联系，这是通过 LangChain 集成完成的。要使用 Jupyter AI，用户可以安装适用于 JupyterLab 的适当版本，并通过聊天界面或魔术命令界面访问它。聊天界面具有 Jupyternaut，一个可以回答问题、解释代码、修改代码和识别错误的 AI 助手。用户还可以从文本提示中生成整个笔记本。该软件允许用户教 Jupyternaut 有关本地文件，并在笔记本环境中使用魔术命令与 LLM 进行交互。它支持多个提供者，并为输出格式提供了定制选项。文档中的此截图显示了聊天功能，Jupyternaut 聊天：

![图 7.2：Jupyter AI – Jupyternaut 聊天。](img/file50.png)

图 7.2：Jupyter AI – Jupyternaut 聊天。

很明显，像这样的聊天工具能方便地提问问题、创建简单函数或更改现有函数，对于数据科学家来说是一种福音。使用这些工具的好处包括提高效率，在模型构建或特征选择等任务中减少手动工作，增强模型的可解释性，识别和修复数据质量问题，与其他 scikit-learn 管道（pandas_dq）集成，以及结果可靠性的整体改进。总的来说，自动化数据科学可以极大加速分析和机器学习应用开发。它让数据科学家集中精力在流程的更高价值和创造性方面。对于商业分析师来说，使数据科学普及化也是自动化这些工作流的一个关键动机。在接下来的章节中，我们将依次讨论这些步骤，并讨论如何自动化它们，以及高效率的生成 AI 如何为工作流的改进做出贡献。

### 数据收集

自动化数据收集是在无需人工干预的情况下进行数据收集的过程。自动数据收集对企业来说是一种有价值的工具。它可以帮助企业更快速、更高效地收集数据，并且可以释放人力资源来专注于其他任务。通常，在数据科学或分析的背景下，我们将 ETL（抽取、转换和加载）视为不仅仅是从一个或多个来源获取数据（数据收集）的过程，还包括准备数据以满足特定用例的过程。ETL 过程通常遵循以下步骤：

1.  提取：数据从源系统中提取出来。这可以通过多种方法来完成，比如网页抓取、API 集成或数据库查询。

1.  转换：数据被转换为数据仓库或数据湖可以使用的格式。这可能涉及数据清洗、去重和标准化数据格式。

1.  载入：数据被载入数据仓库或数据湖中。这可以通过多种方法来完成，比如批量载入或增量载入。

ETL 和数据收集可以使用多种工具和技术来完成，比如：

+   网页抓取：网页抓取是从网站中提取数据的过程。这可以使用多种工具来完成，比如 Beautiful Soup、Scrapy、Octoparse。

+   API（应用程序接口）：这是软件应用程序进行交流的一种方法。企业可以使用 API 从其他公司收集数据，而无需建立自己的系统。

+   查询语言：任何数据库都可以作为数据源，包括 SQL（结构化查询语言）或非 SQL 类型。

+   机器学习：机器学习可以用于自动化数据收集过程。例如，企业可以利用机器学习来识别数据中的模式，然后根据这些模式收集数据。

一旦数据被收集，就可以对其进行处理，以便在数据仓库或数据湖中使用。ETL 过程通常会清理数据、删除重复项并标准化数据格式。然后，数据将被加载到数据仓库或数据湖中，数据分析师或数据科学家可以利用这些数据来获取业务见解。有许多 ETL 工具，包括商业工具如 AWS glue、Google Dataflow、Amazon Simple Workflow Service (SWF)、dbt、Fivetran、Microsoft SSIS、IBM InfoSphere DataStage、Talend Open Studio，或者开源工具如 Airflow、Kafka 和 Spark。在 Python 中有更多的工具，太多了无法列举出来，比如用于数据提取和处理的 Pandas，甚至是 celery 和 joblib，它们可以作为 ETL 编排工具。在 LangChain 中，有与 Zapier 的集成，这是一种可以用于连接不同应用程序和服务的自动化工具。这可以用于自动化来自各种来源的数据收集过程。以下是使用自动化 ETL 工具的一些好处：

+   提高准确性：自动化 ETL 工具可以帮助提高数据提取、转换和加载过程的准确性。这是因为这些工具可以编程遵循一组规则和程序，可以帮助减少人为错误。

+   缩短上市时间：自动化 ETL 工具可以帮助缩短将数据放入数据仓库或数据湖中所需的时间。这是因为这些工具可以自动化 ETL 过程中涉及的重复任务，比如数据提取和加载。

+   提高可伸缩性：自动化的 ETL 工具可以帮助提高 ETL 过程的可伸缩性。这是因为这些工具可以用于处理大量的数据，并且它们可以轻松地根据业务需求进行横向或纵向扩展。

+   改善合规性：自动化 ETL 工具可以帮助改善符合 GDPR 和 CCPA 等法规的程度。这是因为这些工具可以编程遵循一组规则和程序，可以帮助确保数据以符合法规的方式进行处理。

自动化数据收集的最佳工具将取决于企业的具体需求。企业应考虑他们需要收集的数据类型、需要收集的数据量以及他们可用的预算。

### 可视化和 EDA

自动化的探索性数据分析（EDA）和可视化是指利用软件工具和算法自动分析和可视化数据的过程，无需显著的手动干预。传统的探索性数据分析涉及手动探索和总结数据，以了解在执行机器学习或深度学习任务之前的各个方面。它有助于识别模式，检测不一致性，测试假设并获得洞见。然而，随着大型数据集的出现和对高效分析的需求，自动化的 EDA 变得越来越重要。自动化的 EDA 和可视化工具提供了几个好处。它们可以加快数据分析过程，减少在数据清理、处理缺失值、异常值检测和特征工程等任务上花费的时间。这些工具还通过生成交互式可视化来更有效地探索复杂数据集，从而提供对数据的全面概述。有几种工具可用于自动化的 EDA 和可视化，包括：

+   D-Tale：一个库，方便地可视化 pandas 数据框。它支持交互式图形、3D 图形、热图、相关分析、自定义列创建。

+   ydata-profiling（之前是 pandas profiling）：一个生成交互式 HTML 报告（`ProfileReport`）的开源库，总结数据集的不同方面，例如缺失值统计、变量类型分布概况、变量之间的相关性。它可以与 Pandas 以及 Spark DataFrames 一起使用。

+   Sweetviz：一个 Python 库，提供了对探索性数据分析的可视化能力，只需很少的代码。它允许在变量或数据集之间进行比较。

+   Autoviz：该库只需几行代码就可以自动生成各种大小的数据集的可视化。

+   DataPrep：只需几行代码，您就可以从常见的数据源中收集数据，进行 EDA 和数据清理，例如标准化列名或条目。

+   Lux：通过交互式小部件显示数据集中有趣的趋势和模式的一组可视化，用户可以快速浏览以获得洞见。

在数据可视化中使用生成式人工智能为自动化探索性数据分析增加了另一个维度，它允许算法基于现有的可视化结果或特定用户提示生成新的可视化结果。生成式人工智能有潜力通过自动化设计过程的一部分来增强创造力，同时保持对最终输出的人类控制。总体而言，自动化探索性数据分析和可视化工具在时间效率、全面分析和生成有意义的数据可视化方面具有显著优势。生成式人工智能有潜力以多种方式改变数据可视化。例如，它可以用于创建更真实和吸引人的可视化效果，这有助于业务沟通，并更有效地向利益相关者传达数据，以向每个用户提供他们需要获取见解并做出知情决策所需的信息。生成式人工智能可以通过为每个用户量身定制的个性化可视化效果增强和扩展传统工具的创建能力。此外，生成式人工智能可以用于创建交互式可视化效果，允许用户以新颖和创新的方式探索数据。

### 预处理和特征提取

自动化数据预处理是自动化数据预处理任务的过程。 这可以包括诸如数据清洗、数据集成、数据转换和特征提取等任务。 它与 ETL 中的转换步骤相关，因此在工具和技术上有很多重叠。数据预处理很重要，因为它确保数据处于可以被数据分析师和机器学习模型使用的格式。 这包括从数据中删除错误和不一致性，以及将其转换为与将要使用的分析工具兼容的格式。手动工程特征可能很烦琐且耗时，因此自动化此过程非常有价值。 最近，出现了几个开源 Python 库，以帮助从原始数据中自动生成有用的特征，我们将看到。Featuretools 提供了一个通用框架，可以从事务性和关系性数据中合成许多新特征。 它集成了多个 ML 框架，使其灵活。 Feature Engine 提供了一组更简单的转换器，专注于处理缺失数据等常见数据转换。 为了针对基于树的模型专门优化特征工程，来自 Microsoft 的 ta 通过诸如自动交叉等技术表现出强大的性能。AutoGluon Features 将神经网络风格的自动特征生成和选择应用于提高模型准确性。 它与 AutoGluon autoML 功能紧密集成。 最后，TensorFlow Transform 直接在 Tensorflow 管道上运行，以准备模型在训练期间使用的数据。 它已经迅速发展，现在有多种开源选项。Featuretools 提供了最多的自动化和灵活性，同时集成了 ML 框架。 对于表格数据，ta 和 Feature Engine 提供了易于使用的针对不同模型进行优化的转换器。 Tf.transform 非常适合 TensorFlow 用户，而 AutoGluon 专门针对 Apache MXNet 深度学习软件框架。至于时间序列数据，Tsfel 是一个从时间序列数据中提取特征的库。 它允许用户指定特征提取的窗口大小，并可以分析特征的时间复杂性。 它计算统计、频谱和时间特征。另一方面，tsflex 是一个灵活高效的时间序列特征提取工具包，适用于序列数据。 它对数据结构做出了少量假设，并且可以处理缺失数据和长度不等的情况。 它还计算滚动特征。与 tsfresh 相比，这两个库提供了更现代的自动时间序列特征工程选项。 Tsfel 更全面，而 tsflex 强调对复杂序列数据的灵活性。有一些工具专注于机器学习和数据科学的数据质量，带有数据概要文件和自动数据转换。 例如，pandas-dq 库可以与 scikit-learn 管道集成，为数据概要文件、训练测试比较、数据清理、数据填充（填充缺失值）和数据转换（例如，偏斜校正）提供一系列有用的功能。 它通过在建模之前解决潜在问题来改善数据分析的质量。更专注于通过及早识别潜在问题或错误来改进可靠性的工具包括 Great Expectations 和 Deequ。 Great Expectations 是一个用于验证、记录和分析数据的工具，以保持质量并改进团队之间的沟通。 它允许用户对数据断言期望，并通过对数据进行单元测试迅速捕获问题，根据期望创建文档和报告。 Deequ 是建立在 Apache Spark 之上的工具，用于为大型数据集定义数据质量的单元测试。 它让用户明确陈述关于数据集的假设，并通过对属性的检查或约束验证它们。 通过确保遵守这些假设，它可以防止下游应用程序中的崩溃或错误输出。所有这些库都允许数据科学家缩短特征准备时间并扩展特征空间以改进模型质量。自动特征工程对于利用复杂现实世界数据上 ML 算法的全部潜力变得至关重要。

### AutoML

自动化机器学习（AutoML）框架是自动化机器学习模型开发过程的工具。它们可用于自动化诸如数据清洗、特征选择、模型训练和超参数调整等任务。这可以节省数据科学家大量的时间和精力，还可以帮助提高机器学习模型的质量。AutoML 的基本思想在 mljar autoML 库的 Github 仓库中通过这张图解释（来源：https://github.com/mljar/mljar-supervised）：

![图 7.3：AutoML 的工作原理。](img/file51.png)

图 7.3：AutoML 的工作原理。

载入一些数据，尝试不同的预处理方法、ML 算法、训练和模型参数的组合，创建解释，将结果与可视化内容一起在排行榜中进行比较。 AutoML 框架的主要价值主张是易用性和增加开发者在找到机器学习模型、理解它并将其投入生产中的生产力。 AutoML 工具已经存在很长时间了。 最早的广泛框架之一是 AutoWeka，它是用 Java 编写的，并且旨在自动化 Weka（Waikato 知识分析环境）机器学习套件中用于表格数据的机器学习模型开发过程，该套件是在 Waikato 大学开发的。自 AutoWeka 发布以来的几年里，已经开发出了许多其他 AutoML 框架。 如今一些最流行的 AutoML 框架包括 auto-sklearn、autokeras、NASLib、Auto-Pytorch、tpot、optuna、autogluon 和 ray（调整）。 这些框架用多种编程语言编写，并支持多种机器学习任务。自动化机器学习和神经架构搜索的最新进展使得工具能够自动化机器学习管道的大部分内容。 领先的解决方案如 Google AutoML、Azure AutoML 和 H2O AutoML/Driverless AI 可以根据数据集和问题类型自动处理数据准备、特征工程、模型选择、超参数调整和部署。这使得机器学习更容易被非专家所接受。目前的自动机器学习解决方案可以非常有效地处理结构化数据，如表格和时间序列数据。它们可以自动生成相关特征，选择算法，如树集成、神经网络或 SVM，并调整超参数。 由于大量的超参数搜索，性能通常与手动过程相当甚至更好。针对图像、视频和音频等非结构化数据的自动机器学习也正在迅速发展，其中包括神经架构搜索技术。 AutoKeras、AutoGluon 和 AutoSklearn 等开源库也提供了可访问的自动机器学习能力。 但是，大多数自动机器学习工具仍然需要一些编码和数据科学专业知识。 完全自动化数据科学仍然具有挑战性，而且自动机器学习在灵活性和可控性方面存在局限性。 但是，随着更加用户友好和性能更好的解决方案不断问世，进展正在迅速取得。以下是框架的表格摘要：

| **框架** | **语言** | **ML 框架** | **首次发布** | **关键特性** | **数据类型** | **维护者** | **Github 星标** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Auto-Keras | Python | Keras | 2017 | 神经架构搜索，易于使用 | 图像、文本、表格 | Keras 团队（DATA 实验室，德克萨斯农工大学） | 8896 |
| Auto-PyTorch | Python | PyTorch | 2019 | 神经架构搜索，超参数调整 | 表格、文本、图像、时间序列 | AutoML Group，弗莱堡大学 | 2105 |
| Auto-Sklearn | Python | Scikit-learn | 2015 | 自动化的 scikit-learn 工作流 | 表格数据 | 弗赖堡大学 AutoML 小组 | 7077 |
| Auto-WEKA | Java* | WEKA | 2012 | 贝叶斯优化 | 表格数据 | 英属哥伦比亚大学 | 315 |
| AutoGluon | Python | MXNet，PyTorch | 2019 | 专为深度学习优化 | 文本，图像，表格数据 | 亚马逊 | 6032 |
| AWS SageMaker Autopilot | Python | XGBoost，sklearn | 2020 | 基于云的，简单 | 表格数据 | 亚马逊 | - |
| Azure AutoML | Python | Scikit-learn，PyTorch | 2018 | 可解释性模型 | 表格数据 | 微软 | - |
| DataRobot | Python, R | 多种 | 2012 | 监控，可解释性 | 文本，图像，表格数据 | DataRobot | - |
| Google AutoML | Python | TensorFlow | 2018 | 易于使用，基于云的 | 文本，图像，视频，表格数据 | 谷歌 | - |
| H2O AutoML | Python, R | XGBoost，GBMs | 2017 | 自动化工作流，集成学习 | 表格数据，时间序列，图像 | h2o.ai | 6430 |
| hyperopt-sklearn | Python | Scikit-learn | 2014 | 超参数调整 | 表格数据 | Hyperopt 团队 | 1451 |
| Ludwig | Python | Transformers/Pytorch | 2019 | 无代码框架，用于构建和调整自定义 LLM 和深度神经网络 | 多种 | Linux Foundation | 9083 |
| MLJar | Python | 多种 | 2019 | 可解释，可定制 | 表格数据 | MLJar | 2714 |
| NASLib | Python | PyTorch，TensorFlow/Keras | 2020 | 神经结构搜索 | 图像，文本 | 弗赖堡大学 AutoML 小组 | 421 |
| Optuna | Python | 通用 | 2019 | 超参数调整 | 通用 | Preferred Networks 公司 | 8456 |
| Ray (Tune) | Python | 通用 | 2018 | 分布式超参数调整；加速 ML 工作负载 | 通用 | 加州大学伯克利分校 | 26906 |
| TPOT | Python | Scikit-learn，XGBoost | 2016 | 遗传编程，管道 | 表格数据 | Penn State 大学的 Epistasis 实验室 | 9180 |
| TransmogrifAI | Scala | Spark ML | 2018 | 基于 Spark 的 AutoML | 文本，表格数据 | Salesforce | 2203 |

图 7.4：开源 AutoML 框架的比较。Weka 可以通过 pyautoweka 从 Python 访问。Ray Tune 和 H2O 的星号涉及整个项目，而不仅仅是 automl 部分。与 AutoML 相关的 H2O 商业产品是 Driverless AI。大多数项目由一组与任何一个公司无关的贡献者维护

我只列出了最大的框架、库或产品，省略了一些。虽然焦点在于 Python 中的开源框架，但我也包括了一些大的商业产品。Github 星标旨在展示框架的受欢迎程度 - 它们与专有产品无关。Pycaret 是另一个大型项目（7562 颗星），它提供了同时训练多个模型并用相对较少的代码进行比较的选项。像 Nixtla 的 Statsforecast 和 MLForecast，或者是 Darts 这样的项目，具有特定于时间序列数据的类似功能。像 Auto-ViML 和 deep-autoviml 这样的库处理各种类型的变量，分别基于 scikit-learn 和 keras 构建。它们旨在使初学者和专家都能轻松尝试不同类型的模型和深度学习。然而，用户应该行使自己的判断以获得准确和可解释的结果。AutoML 框架的重要功能包括以下内容：

+   部署：有些解决方案，特别是云端解决方案，可以直接部署到生产环境。其他的则导出到 tensorflow 或其他格式。

+   数据类型：大多数解决方案都专注于制表数据集；深度学习自动化框架经常处理不同类型的数据。例如，autogluon 除了制表数据外，还促进了针对图像、文本和时间序列的机器学习解决方案的快速比较和原型制作。像 optuna 和 ray tune 这样专注于超参数优化的几个工具是对格式完全无偏见的。

+   可解释性：这可能非常重要，具体取决于行业，与法规（例如医疗保险）或可靠性（金融）有关。对于一些解决方案，这是一个独特的卖点。

+   监控：部署后，模型性能可能会恶化（漂移）。少数提供者提供性能监视。

+   可访问性：有些提供者需要编码或至少具备基本的数据科学理解，而其他的则是开箱即用的解决方案，几乎不需要编写任何代码。通常，低代码和无代码解决方案的可自定义性较低。

+   开源：开源平台的优点在于它们完全透明地公开了实现和方法及其参数的可用性，并且它们是完全可扩展的。

+   转移学习：这种能力意味着能够扩展或自定义现有的基础模型。

在这里还有很多内容需要涵盖，这将超出本章的范围，比如可用方法的数量。支持较少的功能包括自监督学习、强化学习或生成图像和音频模型。对于深度学习，一些库专注于后端，专门使用 Tensorflow、Pytorch 或 MXNet。Auto-Keras、NASLib 和 Ludwig 具有更广泛的支持，特别是因为它们与 Keras 一起工作。从计划于 2023 年秋季发布的版本 3.0 开始，Keras 支持三个主要的后端 TensorFlow、JAX 和 PyTorch。Sklearn 拥有自己的超参数优化工具，如网格搜索、随机搜索、连续二分法。更专业的库，如 auto-sklearn 和 hyperopt-sklearn，提供了贝叶斯优化的方法。Optuna 可以与各种 ML 框架集成，如 AllenNLP、Catalyst、Catboost、Chainer、FastAI、Keras、LightGBM、MXNet、PyTorch、PyTorch Ignite、PyTorch Lightning、TensorFlow 和 XGBoost。Ray Tune 具有其自身的集成，其中包括 optuna。它们都具有领先的参数优化算法和用于扩展（分布式训练）的机制。除了上述列出的功能外，这些框架中的一些可以自动执行特征工程任务，例如数据清理和特征选择，例如删除高度相关的特征，并以图形方式生成性能结果。每个列出的工具都有它们各自的实现，如特征选择和特征转换的每个步骤-不同之处在于自动化的程度。更具体地说，使用 AutoML 框架的优势包括：

+   时间节约：AutoML 框架可以通过自动化机器学习模型开发的过程，为数据科学家节省大量时间。

+   提高准确性：AutoML 框架可以通过自动化超参数调整的过程来帮助提高机器学习模型的准确性。

+   增强可访问性：AutoML 框架使那些对机器学习经验不多的人更容易接触机器学习。

但是，使用 AutoML 框架也存在一些缺点：

+   黑盒子：AutoML 框架可以是“黑盒子”，这意味着它的工作原理可能难以理解。这可能会使得调试 AutoML 模型的问题变得困难。

+   有限的灵活性：AutoML 框架在能够自动化的机器学习任务类型方面可能会有所限制。

上述工具中有很多都至少具有某种自动特征工程或预处理功能，但是也有一些更专业化的工具可以实现这一点。

### 生成模型的影响

生成式人工智能和像 GPT-3 这样的 LLMs 已经给数据科学和分析领域带来了重大变革。这些模型，特别是 LLMs，有潜力以多种方式彻底改变数据科学的所有步骤，为研究人员和分析师提供令人兴奋的机会。生成式人工智能模型，比如 ChatGPT，能够理解和生成类似人类的回应，成为增强研究生产力的有价值工具。生成式人工智能在分析和解释研究数据方面起着关键作用。这些模型可以协助进行数据探索，发现隐藏的模式或相关性，并提供通过传统方法可能不明显的见解。通过自动化数据分析的某些方面，生成式人工智能节省了时间和资源，使研究人员能够专注于更高级别的任务。生成式人工智能可以在帮助研究人员进行文献综述和识别研究空白方面起到关键作用。ChatGPT 和类似模型可以总结学术论文或文章中的大量信息，提供现有知识的简洁概述。这有助于研究人员更有效地识别文献中的空白并指导他们自己的调查。我们在*第四章* *问题回答*中研究了使用生成式人工智能模型的这一方面。生成式人工智能的其他用例可能包括：

+   自动产生合成数据：生成式人工智能可用于自动生成合成数据，可用于训练机器学习模型。这对于没有大量真实世界数据的企业非常有帮助。

+   识别数据中的模式：生成式人工智能可以用于识别人类分析员无法看到的数据中的模式。这对于希望从数据中获得新见解的企业非常有帮助。

+   从现有数据中创建新特征：生成式人工智能可以用于从现有数据中创建新特征。这对于希望改善他们的机器学习模型准确性的企业非常有帮助。

根据像麦肯锡和毕马威这样的最近报告所述，人工智能的后果涉及数据科学家将工作的内容，他们如何工作以及谁能完成数据科学任务。主要影响包括:

+   人工智能的民主化：生成式模型让更多人通过简单提示生成文本、代码和数据来利用人工智能。这将人工智能的使用扩展到数据科学家以外的范围。

+   提高生产力：通过自动生成代码、数据和文本，生成式人工智能可以加速开发和分析工作流程。这使数据科学家和分析师能够专注于更高价值的任务。

+   数据科学的创新：生成式人工智能正在带来更创新的方式来探索数据，并生成以传统方法不可能的新假设和见解

+   行业的颠覆：生成式人工智能的新应用可能通过自动化任务或增强产品和服务来颠覆行业。数据团队将需要确定高影响力的用例。

+   仍然存在限制：当前模型仍然存在准确性限制、偏见问题和缺乏可控性。需要数据专家监督负责的发展。

+   治理的重要性：对生成式人工智能模型的发展和道德使用进行严格的治理将对维护利益相关者的信任至关重要。

+   合作伙伴关系的需求 - 公司将需要与合作伙伴、社区和平台提供商建立生态系统，以有效利用生成式人工智能的能力。

+   数据科学技能的变化 - 需求可能从编码专业知识转向数据治理、伦理、将业务问题转化为语言以及监督人工智能系统等能力。

关于数据科学的民主化和创新，更具体地说，生成式人工智能也正在影响数据可视化的方式。过去，数据可视化通常是静态的和二维的。然而，生成式人工智能可以用于创建交互式和三维的可视化，这有助于使数据更易于访问和理解。这使得人们更容易理解和解释数据，从而促进更好的决策。同样，生成式人工智能带来的最大变化之一是数据科学的民主化。过去，数据科学是一个非常专业化的领域，需要对统计学和机器学习有深入的理解。然而，生成式人工智能使得不具备较高技术专业知识的人们能够创建和使用数据模型。这使得数据科学领域对更广泛的人群开放。LLMs 和生成式人工智能可以在自动化数据科学中发挥关键作用，提供几个优势：

+   自然语言交互：LLMs 允许进行自然语言交互，使用户能够使用普通英语或其他语言与模型进行交流。这使得非技术用户可以使用日常语言与数据进行交互和探索，而无需具备编码或数据分析方面的专业知识。

+   代码生成：生成式人工智能可以自动生成代码片段，以执行探索性数据分析期间的特定分析任务。例如，它可以生成检索数据的代码（例如，SQL）、清理数据、处理缺失值或创建可视化（例如，在 Python 中）。此功能节省时间，减少了手动编码的需求。

+   自动报告生成：LLM（大型语言模型）可以生成自动化报告，总结探索性数据分析的关键发现。这些报告提供了关于数据集各个方面的见解，例如统计摘要、相关性分析、特征重要性等，使用户更容易理解和展示他们的发现。

+   数据探索和可视化：生成式人工智能算法可以全面地探索大型数据集，并自动生成可视化图表，揭示数据中的基本模式、变量之间的关系、离群值或异常值。这有助于用户全面理解数据集，而无需手动创建每个可视化图表。

此外，我们可以认为生成式 AI 算法应该能够从用户交互中学习，根据个人偏好或过去行为调整推荐内容。它们通过持续自适应学习和用户反馈而不断优化，为自动化 EDA 提供更加个性化和有用的见解。最后，生成式 AI 模型可以通过从现有数据集中学习模式（智能错误识别）在 EDA 过程中识别数据中的错误或异常。它们能够快速准确地检测不一致性并突出潜在问题。总的来说，LLM 和生成式 AI 可以通过简化用户互动、生成代码片段、高效识别错误/异常、自动化报告生成、实现全面数据探索和可视化建设以及适应用户偏好来增强自动化 EDA，以便更有效地分析大型和复杂数据集。然而，虽然这些模型可以极大地增强研究和文献综述过程，但它们不应被视为绝对可靠的来源。如前所述，LLM 是通过类比工作的，在推理和数学方面会遇到困难。它们的优势在于创造力，而不是准确性，因此研究人员必须运用批判性思维，并确保这些模型生成的输出准确、公正并符合严格的科学标准。其中一个著名的例子是微软的 Fabric，它包含由生成式 AI 驱动的聊天界面，使用户能够使用自然语言提出数据相关问题，并在无需等待数据请求队列的情况下立即获得答案。通过利用像 OpenAI 模型这样的 LLM，Fabric 实现了对有价值的见解的实时访问。Fabric 在其他分析产品中脱颖而出，因为它采用全面的方法。它解决了组织在分析过程中各个方面的需求，并为参与分析过程的不同团队（如数据工程师、数据仓库专业人员、科学家、分析人员和业务用户）提供角色专用的体验。借助每个层面的 Azure OpenAI 服务的集成，Fabric 利用生成式 AI 的能力来发掘数据的全部潜力。类似 Microsoft Fabric 中的 Copilot 等特性提供了对话式语言体验，使用户能够创建数据流、生成代码或整个函数、构建机器学习模型、可视化结果，甚至开发自定义的对话式语言体验。有趣的是，ChatGPT（以及它的扩展 Fabric）经常会产生不正确的 SQL 查询。虽然对于可以检查输出有效性的分析人员来说这没问题，但对于非技术业务用户而言，这是一场灾难性的自助式分析工具。因此，在使用 Fabric 进行分析时，组织必须确保有可靠的数据管道，并采取数据质量管理实践。虽然生成式 AI 在数据分析中的潜力很大，但仍需要谨慎。必须通过第一原理推理和严格的分析来验证 LLM 的可靠性和准确性。尽管这些模型在临时分析、研究中的思想生成和复杂分析的概括方面表现出了其潜力，但由于需要领域专家的验证，它

## 代理人可以回答数据科学问题

正如我们在 Jupyter AI（Jupyternaut chat）中所看到的 - 还有第六章的*开发软件* - 通过生成式 AI（代码 LLMs）来增加创建和编写软件的效率有很大的潜力。这是我们研究数据科学中使用生成式 AI 的实际部分的很好的起点。我们之前已经看到不同的带有工具的代理人。例如，LLMMathChain 可以执行 Python 来回答数学问题，就像这里所示：

```py
from langchain import OpenAI, LLMMathChain
llm = OpenAI(temperature=0)
llm_math = LLMMathChain.from_llm(llm, verbose=True)
llm_math.run("What is 2 raised to the 10th power?")
```

尽管这对于提取信息并将其反馈是有用的，但如何将其插入传统的 EDA 过程中却不太明显。同样，CPAL(`CPALChain`)和 PAL(`PALChain`)链可以回答更复杂的推理问题，同时保持幻觉受控，但很难想出它们的真实用例。通过`PythonREPLTool`，我们可以使用玩具数据创建简单的可视化，或者用合成数据进行训练，这对说明或启动项目可能很好。这是 LangChain 文档中的一个例子：

```py
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
agent_executor = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000),
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
agent_executor.run(
    """Understand, write a single neuron neural network in PyTorch.
Take synthetic data for y=2x. Train for 1000 epochs and print every 100 epochs.
Return prediction for x = 5"""
)
```

请注意，这应谨慎执行，因为 Python 代码直接在机器上执行而且没有任何防护措施。实际上这是有效的，可以创建数据集，训练模型，然后得到预测结果：

```py
Entering new AgentExecutor chain...
I need to write a neural network in PyTorch and train it on the given data
Action: Python_REPL
Action Input: 
import torch
model = torch.nn.Sequential(
    torch.nn.Linear(1, 1)
)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Define the data
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
for epoch in range(1000):  # Train the model
    y_pred = model(x_data)
    loss = loss_fn(y_pred, y_data)
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}: {loss.item():.4f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# Make a prediction
x_pred = torch.tensor([[5.0]])
y_pred = model(x_pred)
Observation: Epoch 100: 0.0043
Epoch 200: 0.0023
Epoch 300: 0.0013
Epoch 400: 0.0007
Epoch 500: 0.0004
Epoch 600: 0.0002
Epoch 700: 0.0001
Epoch 800: 0.0001
Epoch 900: 0.0000
Epoch 1000: 0.0000
Thought: I now know the final answer
Final Answer: The prediction for x = 5 is y = 10.00.
```

再次，这非常酷，但很难看出如何在没有更严谨的工程的情况下扩展。如果我们想要丰富我们的数据以获取类别或地理信息，LLMs 和工具会很有用。例如，如果我们的公司从东京提供航班，而我们想要知道我们的客户距离东京的距离，我们可以使用 Wolfram Alpha 作为一个工具。这是一个简单的例子：

```py
from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
llm = OpenAI(temperature=0)
tools = load_tools(['wolfram-alpha'])
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory, verbose=True)
agent.run(
    """How far are these cities to Tokyo?
* New York City
* Madrid, Spain
* Berlin
""")
```

请确保您已经设置了 OPENAI_API_KEY 和 WOLFRAM_ALPHA_APPID 环境变量，正如在第三章*开始使用 LangChain*中所讨论的那样。这是输出结果：

```py
> Entering new AgentExecutor chain...
AI: The distance from New York City to Tokyo is 6760 miles. The distance from Madrid, Spain to Tokyo is 8,845 miles. The distance from Berlin, Germany to Tokyo is 6,845 miles.
> Finished chain.
'
The distance from New York City to Tokyo is 6760 miles. The distance from Madrid, Spain to Tokyo is 8,845 miles. The distance from Berlin, Germany to Tokyo is 6,845 miles.
```

现在，很多这些问题都非常简单。然而，我们可以给予代理人数据集进行处理，这就是当我们连接更多工具时，它变得非常强大的地方。让我们开始问答关于结构化数据集的问题吧！

## 使用 LLMs 进行数据探索

数据探索是数据分析中至关重要且基础的步骤，使研究人员能够全面了解其数据集并发现重要信息。随着类似 ChatGPT 这样的 LLM 的出现，研究人员可以利用自然语言处理的能力促进数据探索。正如我们之前提到的生成式 AI 模型如 ChatGPT 具有理解和生成类人回答的能力，使它们成为增强研究生产力的有价值的工具。以自然语言提出问题并获得易消化的响应可以极大地促进分析。LLM 不仅可用于探索文本数据，还可用于探索其他形式的数据，如数字数据集或多媒体内容。研究人员可以利用 ChatGPT 的能力，询问数值数据集中的统计趋势或查询图像分类任务的可视化。让我们加载一个数据集并进行处理。我们可以从 scikit-learn 快速获取一个数据集：

```py
from sklearn.datasets import load_iris
df = load_iris(as_frame=True)["data"]
```

Iris 数据集是众所周知的-它是玩具数据集，但它将帮助我们说明使用生成式 AI 进行数据探索的能力。我们将在接下来使用 DataFrame。我们现在可以创建一个 Pandas dataframe 代理，看看如何轻松地完成一些简单的事情！

```py
from langchain.agents import create_pandas_dataframe_agent
from langchain import PromptTemplate
from langchain.llms.openai import OpenAI
PROMPT = (
    "If you do not know the answer, say you don't know.\n"
    "Think step by step.\n"
    "\n"
    "Below is the query.\n"
    "Query: {query}\n"
)
prompt = PromptTemplate(template=PROMPT, input_variables=["query"])
llm = OpenAI()
agent = create_pandas_dataframe_agent(llm, df, verbose=True)
```

我已经为模型制定了当怀疑时告诉它自己不知道以及逐步思考的指示，这两者都可以减少产生幻觉的可能性。现在我们可以针对 DataFrame 查询我们的代理：

```py
agent.run(prompt.format(query="What's this dataset about?"))
```

我们得到了答案 "这个数据集是关于某种类型的花的测量"，是正确的。让我们展示如何获得一个可视化：

```py
agent.run(prompt.format(query="Plot each column as a barplot!"))
```

它并不完美，但我们得到了一个好看的图表：

![图 7.5：Iris 数据集的条形图。](img/file52.png)

图 7.5: Iris 数据集的条形图。

我们也可以请求以可视化方式查看列的分布，从而获得这个整洁的图表：

![图 7.6：Iris 数据集箱线图。](img/file53.png)

图 7.6: Iris 数据集箱线图。

我们可以请求图表使用其他绘图后端，例如 seaborn，但请注意，这些必须安装。我们还可以询问关于数据集的更多问题，比如哪一行在花瓣长度和花瓣宽度之间有最大差异。我们得到了具有中间步骤的答案（缩短后）。

```py
df['difference'] = df['petal length (cm)'] - df['petal width (cm)']
df.loc[df['difference'].idxmax()]
Observation: sepal length (cm)    7.7
sepal width (cm)     2.8
petal length (cm)    6.7
petal width (cm)     2.0
difference           4.7
Name: 122, dtype: float64
Thought: I now know the final answer
Final Answer: Row 122 has the biggest difference between petal length and petal width.
```

我认为这值得称赞，LLM！下一步可能是给提示添加更多关于绘图的指示，例如绘图大小等。在 Streamlit 应用程序中实现相同的绘图逻辑有点困难，因为我们需要使用适当的 Streamlit 函数的绘图功能，例如`st.bar_chart()`，但是这也可以完成。您可以在 Streamlit 博客上找到有关此内容的解释（“使用 ChatGPT 构建 Streamlit 和 scikit-learn 应用程序”）。那么统计测试呢？

```py
agent.run(prompt.format(query="Validate the following hypothesis statistically: petal width and petal length come from the same distribution."))
```

我们得到了这个回答：

```py
Thought: I should use a statistical test to answer this question.
Action: python_repl_ast 
Action Input: from scipy.stats import ks_2samp
Observation: 
Thought: I now have the necessary tools to answer this question.
Action: python_repl_ast
Action Input: ks_2samp(df['petal width (cm)'], df['petal length (cm)'])
Observation: KstestResult(statistic=0.6666666666666666, pvalue=6.639808432803654e-32, statistic_location=2.5, statistic_sign=1)
Thought: I now know the final answer
Final Answer: The p-value of 6.639808432803654e-32 indicates that the two variables come from different distributions.
```

'6.639808432803654e-32 的 p 值表明两个变量来自不同的分布。'这是统计检验！很酷。我们可以用简单的提示用普通英语询问关于数据集的相当复杂的问题。还有 pandas-ai 库，它在内部使用 LangChain 并提供类似的功能。以下是文档中的一个例子，案例数据集：

```py
import pandas as pd
from pandasai import PandasAI
df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
    "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
})
from pandasai.llm.openai import OpenAI
llm = OpenAI(api_token="YOUR_API_TOKEN")
pandas_ai = PandasAI(llm)
pandas_ai(df, prompt='Which are the 5 happiest countries?') 
```

当我们直接使用 LangChain 时，这将给我们提供所请求的结果。请注意，pandas-ai 并不是本书的设置的一部分，所以如果你想使用它，你需要单独安装它。对于 SQL 数据库中的数据，我们可以使用`SQLDatabaseChain`进行连接。LangChain 的文档展示了这个例子：

```py
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
db = SQLDatabase.from_uri("sqlite:///../../../../notebooks/Chinook.db")
llm = OpenAI(temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
db_chain.run("How many employees are there?")
```

我们首先连接到数据库。然后我们可以用自然语言提出关于数据的问题。这也可以非常强大。LLM 将为我们创建查询。我期望当我们不了解数据库模式时，这将特别有用。`SQLDatabaseChain`还可以在`use_query_checker`选项设置时检查查询并自动更正它们。让我们做个总结！

## 摘要

在本章中，我们探讨了自动化数据分析和数据科学的最新技术。有很多领域，LLM 可以使数据科学受益，主要是作为编码助手或数据探索。我们从概述了覆盖数据科学流程中每个步骤的框架开始，比如 AutoML 方法，然后讨论了 LLM 如何帮助我们进一步提高生产力，使数据科学和数据分析更加容易访问，无论是对利益相关者还是对开发人员或用户。我们随后研究了代码生成和类似代码 LLM *第六章* *软件开发*中的工具如何在数据科学任务中帮助我们通过创建我们可以查询的函数或模型，或者如何利用 LLM 或第三方工具如沃尔夫拉姆阿尔法来丰富数据。然后，我们关注了在数据探索中使用 LLM。在*第四章* *问答*中，我们研究了摄取大量文本数据以进行分析。在本章中，我们聚焦于 SQL 或表格形式的结构化数据集的探索性分析。总之，人工智能技术有潜力彻底改变我们分析数据的方式，ChatGPT 插件或微软 Fabric 就是例子。然而，在当前的状况下，人工智能不能取代数据科学家，只能帮助他们。让我们看看你是否记住了本章的一些关键要点！

## 问题

请看看你是否能够凭记忆回答这些问题。如果对任何问题不确定，我建议您回到本章的相应部分查看：

1.  数据科学和数据分析之间有什么区别？

1.  数据科学涉及哪些步骤？

1.  为什么我们想要自动化数据科学/分析？

1.  存在用于自动化数据科学任务的框架以及它们能做什么？

1.  生成式人工智能如何帮助数据科学家？

1.  我们可以使用什么样的代理和工具来回答简单的问题？

1.  如何让 LLM 处理数据？
