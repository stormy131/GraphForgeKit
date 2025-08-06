# Enhancer system

**Note:** This repository is a copy of my bachelor thesis. The user guide below is only a part of its text. The full thesis can be found at [here](linkhttps://is.cuni.cz/studium/dipl_st/index.php?id=b8580bff7674eb2b78bd0daccbf29a76&tid=1&do=xdownload&fid=130436322&did=278796&vdetailu=1).

The `Enhancer` system is designed to support both technical and non-technical users through two modes of operation. It can be used as a Python package by developers, allowing direct access to its internal components and customization of its behavior through code. For users without programming experience, the system also supports a standalone application mode, where all behavior is controlled through a structured configuration file. This dual-mode setup makes the system both flexible and accessible, allowing a wide range of users to apply it to their data without sacrificing control or robustness.

At its core, the `Enhancer` addresses the task of tabular data enhancement—reformulating independent data instances into a relational structure that captures meaningful inter-instance relationships. Traditional tabular models, such as MLPs, treat each data point independently and cannot exploit any proximity or similarity between them. The `Enhancer` mitigates this limitation by transforming tabular datasets into graphs, where nodes represent data instances and edges encode inferred relationships between them.

The relationships are defined by a set of configurable *edge creation strategies*, which determine how edges should be formed based on the spatial (or otherwise relational) features present in the dataset. Multiple strategies are supported, ranging from distance-based methods like $K$-Nearest Neighbors and thresholding to structure-driven strategies such as Anchor or Grid partitioning.

Once the graph is constructed, the data can be passed to a GNN model, which combines the relational structure with traditional learning pipelines. Specifically, the GNN augments the fixed MLP estimator with a graph-based *Encoder* that propagates information through the graph using *message-passing* operations. This approach allows the model to utilize both local instance features and the broader relational context, which can improve predictive performance in cases where inter-instance dependencies are meaningful.

This manual provides detailed instructions on how to configure and use the system in both modes of operation, including descriptions of supported strategies, model settings, and usage workflows.

This guide is structured into three main parts:

- [Prerequisites](#prerequisites) outlines the necessary preparatory steps that must be completed before the user can begin using the Enhancer.
- [Application-based mode configuration](#application-based-mode-configuration) describes a non-expert user-oriented mode based on configuration files and application-level usage.
- [Developer mode configuration](#developer-mode-configuration) presents the developer-oriented inference mode, highlighting the flexibility and extensibility available through direct integration.

## Prerequisites

While the configuration approach described in this chapter is designed to minimize the need for programming expertise, a minimal level of technical preparation is still required to ensure that the Enhancer system runs correctly. Before performing any inference tasks, users must first set up the appropriate execution environment. This involves two key steps. First, it is essential to have Python installed in the system, with a minimum required version of 3.10. This ensures compatibility with all features and libraries used within the Enhancer system. Second, the user is advised to prepare a Python virtual environment dedicated to the project. This can be accomplished by installing the required dependencies listed in the `requirements.txt` file, which accompanies the project repository. Executing this step guarantees that all libraries and tools needed by the Enhancer are properly configured and available within the environment.

An example of a typical setup procedure is shown in the bash listing below:

```bash
# Check Python version (should be 3.10 or higher)
python3 --version

# Create a new virtual environment "enhancer-env"
python3 -m venv enhancer-env

# Activate the virtual environment (Linux/macOS)
source enhancer-env/bin/activate

# On Windows (PowerShell)
.\enhancer-env\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

Completing these two steps ensures that the system is ready for use, providing a stable foundation for executing both transformation and comparison tasks through the JSON-based configuration interface.

## Application-based mode configuration

To accommodate users who may not wish or are unable to configure the `Enhancer` system through direct Python programming, an alternative inference option has been developed. This user-friendly interface enables system configuration through a structured `JSON` file. The use of `JSON` ensures a standardized, human-readable format for specifying all necessary parameters for system inference, whether in transformation or comparison mode. Internally, the application-based inference reduces to the `Enhancer.process_tasks()` static method. This method executes the full processing pipeline for each pair of dataset and strategy independently, rather than comparing multiple strategies on a shared dataset split. This design allows users to specify and evaluate distinct enhancement tasks without needing to manage train-test partitions or control the pipeline programmatically.

### Entry point for inference execution

The application-based configuration approach treats the Enhancer system as a standalone application, as opposed to the library-style usage typically employed by developers. To facilitate this application-level usage, the `Enhancer` package includes a dedicated entry-point script, `main.py`. This script acts as the main interface for initiating inference tasks via the command line. Users are expected to execute this script from a terminal, supplying it with the necessary command-line arguments that control the behavior and configuration of the system. The available flags are as follows:

- `-m` / `–mode`: Specifies the inference mode of the Enhancer system. Valid options are `transform` for applying data transformations and `compare` for evaluating and comparing different edge-creation strategies. If not explicitly provided, the system defaults to the `compare` mode.
- `-c` / `–config-file`: Indicates the path to the configuration file, which must be in JSON format. This path can be specified either as an absolute location or relative to the location of the `main.py` script.
- `-i` / `–input-data`: Defines the path to the input dataset on which the Enhancer system will operate. The system expects a CSV file (values separated by commas) without index columns and column headers.
- `-o` / `–output-dir`: Sets the output directory where the transformed data will be saved. This flag is applicable only in `transform` mode and is ignored otherwise.

For example, respective `transform` or `compare` procedures can be run as follows:

```bash
# in the root of the Enhancer project

# Starting transformation of the input dataset, based on setup from configuration file
python3 src/main.py -m transform -c ./config1.json -i ./data/dataset.csv -o ./data/outputs

# Compare dataset transformations, listed in the configuration file 
python3 src/main.py -m compare -c ./config2.json -i ./data/dataset.csv
```

Once the appropriate flags are set and the script is executed, the system proceeds to perform either transformation or comparison tasks according to the selected mode and the configuration parameters provided in the JSON file. The resulting output is then written either to the standard output (in the case of comparison reports) or to the specified output directory (in the case of dataset transformation).

### Configuration file format

In application mode, the system accepts a `JSON` configuration file that specifies all necessary parameters for model training and strategy evaluation. This file follows a structured schema and includes definitions for the overall task type, GNN architecture, and a list of strategy configurations to apply. File structure and contents will be validated by the system before the actual pipeline execution, and the user will be notified in case if the configuration does not follow the defined schema. Below, we present the essential fields of the configuration file, detailing their valid values and intended functionality.

#### Top-level fields

- **`problem_type`**: Specifies the prediction objective. Must be either "regression" or "classification". This choice controls which evaluation metrics are used internally and how targets are interpreted. See more details on predictive task type configuration in the [Task Type Configuration](#task-type-configuration) section.
- **`gnn_config`**: A dictionary specifying the architecture of the GNN model. See detailed structure in the [GNN Architecture](#gnn-architecture-gnn_config-field) section below.
- **`tasks`**: A list of strategy configurations, each describing one data-to-graph transformation task using a different edge creation strategy. See detailed structure in the [List of Strategy Configurations](#list-of-strategy-configurations-tasks-field) section below.

#### GNN architecture (`gnn_config` field)

This section defines the neural network architecture for both the Encoder (graph convolutional layers) and the Estimator (dense layers applied to node embeddings) parts of the GNN:

- **`encoder_schema`**: A list of integers specifying the number of units in each graph convolutional layer of the encoder. For example, `[256, 256]` creates two layers with 256 output dimensions each.
- **`estimator_schema`**: A list of integers specifying the number of units in each fully connected layer of the Estimator (MLP). The final number must match the output dimension required by the task (e.g., $1$ for scalar regression, $C$ for classification with C classes).
- **`convolution`**: A string indicating the graph convolution operator to use. The value must be one of the supported convolution types listed in the [Available Convolutions](#available-convolutions) section. If an unsupported operator is provided, validation fails.

#### List of strategy configurations (`tasks` field)

Each entry in the `tasks` list corresponds to one run of the Enhancer pipeline with a specific edge creation strategy:

- **`type`**: A string naming the edge creation strategy. This value must match one of the available strategy identifiers listed in the [Strategy Configuration](#strategy-configuration) section. If an unsupported option is provided, validation fails.
- **`spatial_idx`**: A list of integers indicating the column indices of the spatial features within the processed dataset. These features are used by the strategy to define inter-instance relations.
- **`target_idx`**: An integer specifying the index of the target column in the dataset. This field is required for model training and performance evaluation.
- **`kwargs`**: A dictionary of additional parameters passed to the selected strategy. These are strategy-specific and control edge creation behavior. You can find a set of accepted parameters for each of the built-in strategies in the [Strategy Configuration](#strategy-configuration) section.

An example configuration file content is shown below:

```json
{
    "task_type": "regression", 

    "gnn_config": {
        "encoder_schema":   [256, 256],
        "estimator_schema": [128, 128, 1],
        "convolution":      "SAGE"
    },

    "tasks": [
        {
            "type": "Anchor",
            "spatial_idx": [1, 2],
            "target_idx": 0,

            "kwargs": {
                "n_clusters": 50,
                "cluster_sample_rate": 0.4,
                "cache_id": "melbourne_anchor",
                "cache_dir": "./data/cache/edges"
            }
        },

        {
            "type": "KNN",
            "spatial_idx": [1, 2],
            "target_idx": 0,

            "kwargs": {
                "K": 10,
                "dist_metric": "Euclidean",
                "cache_id": "melbourne_knn",
                "cache_dir": "./data/cache/edges"
            }
        }
    ]
}
```

This configuration format allows for extensive control over the modeling and transformation process, enabling users to test multiple strategies and network architectures with minimal manual effort. All configuration fields are validated before execution to ensure correctness.

## Developer mode configuration

For typical developer use cases, the proposed solution offers a flexible and extensible framework that can be broadly adopted for comparing various edge-creation mechanisms and performing data transformations. The core functionality is encapsulated in the `Enhancer` class, which is accompanied by several built-in components (described in the [Resources Appendix](#resources-appendix)) packaged for general-purpose use. Each supported use case follows a common preparation procedure, which must be completed to ensure compatibility between the user-defined configuration and the system's internal processing pipeline. These preparation steps establish a standardized input format and configuration interface, enabling seamless integration of custom edge-building strategies, neural architectures, and evaluation workflows. The typical workflow consists of the following steps:

### Data preparations

The first step involves formatting the input data to meet the expected structure. The package provides a predefined data structure, `EnhancerData`, which is implemented as a Python `NamedTuple`. This type acts as a formal contract specifying the expected input format. It contains three attributes:

- `features`: a matrix of shape *($N_\textnormal{samples}, N_\textnormal{features}$)* representing node-level feature vectors,
- `target`: a vector of shape *($N_\textnormal{samples},$)* denoting target labels or regression values,
- `spatial`: a matrix of shape *($N_\textnormal{samples}, N_\textnormal{spatial features}$)* encoding the spatial component of each data sample.

### Defining candidate edge-creation strategies

The next step involves specifying the edge-creation strategies to be evaluated. The user may choose from several built-in edge creation strategies, each implementing a different heuristic for constructing edges between data instances based on their spatial characteristics. These strategies serve as practical defaults and are well-suited for experimentation. They are described below:

- **KNN (K-Nearest Neighbors)**: Constructs edges by connecting each data point to its $K$ spatially nearest neighbors. This strategy ensures local connectivity and is sensitive to the density of the dataset. The parameter $K$ controls the number of neighbors per node.
- **Threshold**: Connects all pairs of points whose pairwise distance is below a specified threshold. This method allows for more flexible connectivity patterns, where the scale of the threshold influences the edge density. It can lead to highly connected graphs in dense regions.
- **Anchor**: First partitions the dataset by selecting a fixed number of representative points (anchors) using clustering. Data points are then connected to nearby anchors, and optionally to other points within the same cluster. This approach introduces a hierarchical structure to the graph, reflecting both local and cluster-level relationships.
- **Grid**: Divides the input space into a regular grid of fixed-size bins. Points within the same grid cell are connected (intra-cell), and selected points from each cell are connected to neighboring cells (inter-cell). This strategy is particularly effective when the data exhibits spatial locality aligned with a grid-like layout.

A detailed explanation of each strategy, including an overview of its underlying mechanism, can be found in the [Strategies](#strategies) section. Configuration details and parameter descriptions for each strategy are provided in the [Resources Appendix](#resources-appendix).

Alternatively, the package includes an abstract base class, `BaseStrategy`, which serves as the foundation for implementing custom strategies. Custom implementations must subclass `BaseStrategy` and implement its `__call__` method, which takes a feature matrix and returns an edge index. The returned index must have shape *(2, N_edges)*, where each edge is represented as a column of `[source_idx, destination_idx]` node indices. This format—commonly known as the Coordinate (COO) format—is required by the `PyG` framework, which is used internally for graph representation and modeling. To avoid redundant data transformations and ensure seamless integration, the same format is used throughout the system.

In addition, a separate `utils` submodule provides a set of distance metrics used as core building blocks within edge strategies. These include implementations of common metrics (e.g., Euclidean, cosine) and are designed to be reusable and composable. Detailed description of the metrics can be found in the [Metrics](#metrics) section.

It is important to note that when implementing custom instances of the `BaseStrategy` class, developers must ensure compatibility between the shape of the spatial data and the internal logic of their edge-building algorithm. The built-in implementations, however, are agnostic to the specific spatial dimensions. These implementations treat each spatial vector as a generic embedding and rely solely on the relative structure among samples.

Additionally, custom strategy implementations are required to call the constructor of the `BaseStrategy` superclass with appropriate `cache_dir` and `cache_id` parameters. This step enables the caching mechanism within the `Enhancer` system, which allows reuse of previously computed edge indices for repeated calls with the same identifier. The `cache_dir` parameter specifies the directory where cache files are stored, while `cache_id` serves as a unique identifier for the cached result.

```python
import numpy as np
from torch import Tensor

from strategies import BaseStrategy

class CustomStrategy(BaseStrategy):
    def __init__(self, *some_arguments):
        # required parent constructor call
        super().__init__(
            cache_dir="./cache_data",
            cache_id="my_strategy",
        )

    def __call__(self, data: Tensor) -> Tensor:
        # your implementation of the strategy 
        ...
```

### Specifying the GNN Architecture

The final configuration step involves defining the architecture of the GNN used to evaluate edge-creation strategies. To ensure fair and meaningful comparisons, all strategies should be tested using the same GNN structure.

The system provides a `NetworkConfig` object, implemented as a Python `NamedTuple`, to configure this architecture. It consists of two components:

- `encoder`: a list of tuples of the form `(layer, signature)`. Each tuple represents a graph convolution or transformation layer along with its forward signature (as required by the `PyG` API). These layers are responsible for encoding node features using message passing.
- `estimator`: a standard list of `PyTorch` modules, defining the prediction part of the model that operates after feature encoding.

The **layer signature** is a string that specifies which input components are passed to each layer during forward propagation. Internally, the system supports two predefined components:

- `x`: the node feature matrix,
- `edge_index`: the edge list in COO format.

The user should define the signature accordingly, based on what the layer expects. For example:

- For a `SAGEConv` layer that consumes both `x` and `edge_index`, the correct signature is `"x, edge_index -> x"`.
- For a standard non-graph layer like `Dropout` or `ReLU`, which operates only on `x`, the signature is `"x -> x"`.

This design ensures compatibility with `PyG`'s functional composition and allows flexible modeling of message-passing networks. In contrast, the estimator follows a regular `PyTorch` sequential layout and does not require layer signatures. The example below demonstrates how to define a GNN configuration using this interface:

```python
from torch.nn import ReLU, Dropout, Linear
from torch_geometric.nn import SAGEConv

from enhancer import Enhancer
from schema.configs import NetworkConfig
from schema.data import EnhancerData

# processed dataset (simplified)
data = EnhancerData(...)

# define Encoder part: each layer paired with a signature
test_encoder = [
    (
        SAGEConv(data.features.shape[1], 256),
        "x, edge_index -> x"
    ),
    (Dropout(p=0.3), "x -> x"),
    (SAGEConv(256, 256), "x, edge_index -> x"),
    (Dropout(p=0.3), "x -> x"),
]

# define Estimator part
test_estimator = [
    Linear(256, 128),
    ReLU(),
    Linear(128, 128),
    ReLU(),
    Linear(128, 1),
]

# create GNN configuration
gnn_setup = NetworkConfig(
    encoder=test_encoder,
    estimator=test_estimator,
)
```

### Training configuration

In developer mode, the training procedure can be customized through the `TrainConfig` class, located in the `schema.configs` module. This class serves as a centralized container for all relevant training hyperparameters and allows users to define their training setup before passing it to the methods `Enhancer.compare_strategies()` or `Enhancer.process_tasks()`.

To use it, the user must create an instance of `TrainConfig`, configure the desired parameters, and include it in the execution pipeline. The available parameters, expected type, and their roles are listed below:

- `n_epochs (int)` — Specifies the total number of training epochs. This determines the number of times the model will iterate over the full training dataset.
- `learn_rate (float)` — Defines the learning rate used by the optimizer. It controls the size of parameter updates during training.
- `loss_criteria (torch.nn.Module)` — A `PyTorch` loss function (e.g., `torch.nn.MSELoss()`) that defines the objective to be minimized during training.
- `batch_size (int)` — Sets the number of samples in each training batch. Larger batch sizes may reduce training noise but require more memory.
- `node_vicinity (list[int])` — Specifies the number of neighboring nodes to include at each level of neighborhood sampling during mini-batch training. Each element in the list represents the number of neighbors to sample at a given depth, enabling efficient training on large graphs by operating on subgraphs rather than the entire graph at once.
- `val_ratio (float)` — Determines the proportion of the dataset to be allocated for validation. This is used to monitor model performance during training.
- `test_ratio (float)` — Sets the proportion of the dataset to reserve for final evaluation after training.

## Enhancer public API

The `Enhancer` class provides the main interface for applying graph-based enhancement to tabular datasets and comparing different edge-creation strategies. It supports both single-run workflows and batch experimentation. Below, we describe its public methods, expected parameters, and their functionality.

### Instance Initialization

An `Enhancer` instance is constructed using:

- `gnn_config` (`NetworkConfig`): Specifies the structure of the graph neural network, including both encoder and estimator layers.
- `train_config` (`TrainConfig`): Contains training parameters such as the number of epochs, batch size, and learning rate.
- `strategy` (`BaseStrategy`): Defines how the edge index is constructed from spatial features.

This combination defines how data is transformed, modeled, and trained.

### `fit(data)`

Trains a GNN model using the specified edge-creation strategy and returns both the trained model and the processed graph.

**Parameters**:

- `data` (`EnhancerData`): The dataset to train on, containing node features, targets, and spatial coordinates.

**Returns**:

- Trained `GNN` object.
- `Data` object from [PyG](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html), representing the constructed graph used in training.

### `transform(data)`

Applies the trained GNN encoder to a dataset to produce transformed node features.

**Parameters**:

- `data` (`EnhancerData`): Dataset to transform.

**Returns**:

- `NumPy` array with transformed node representations.

**Note**: This method must be called after `fit()` has been executed. Otherwise, the encoder is undefined.

### `process_tasks(gnn_config, train_config, tasks)`

Runs multiple independent enhancement tasks defined by (strategy, dataset) pairs. Each task creates a new instance of `Enhancer` and applies its own internal train/validation/test split.

**Parameters**:

- `gnn_config` (`NetworkConfig`): GNN architecture used in all tasks.
- `train_config` (`TrainConfig`): Training settings shared across tasks.
- `tasks` (Iterable of `Task`): Each task is a tuple consisting of a strategy and a dataset.

**Returns**:

- A `RunReporter` instance summarizing all completed tasks.

**Note**: As each task is handled separately, the data splits will differ across tasks, making this method suitable for exploratory or loosely coupled experiments.

### `compare_strategies(...)`

Evaluates and compares multiple strategies on a shared dataset split. This method ensures consistency by fixing the same training and testing partitions across all strategies.

**Parameters**:

- `train_data` (`EnhancerData`): Training set.
- `test_data` (`EnhancerData`): Evaluation set.
- `gnn_config` (`NetworkConfig`): GNN definition.
- `train_config` (`TrainConfig`): Training settings.
- `strategies` (list of `BaseStrategy`): Strategies to be tested.

**Returns**:

- A `RunReporter` instance with results from all runs.

**Note**: This method is ideal for fair comparisons as it guarantees identical data conditions for every run.

## Reporter API

The `RunReporter` class provides tools for aggregating, analyzing, and visualizing the outcomes of multiple enhancement runs. Each run corresponds to a model execution using a specific edge-creation strategy. The reporter allows the user to compare prediction performance and structural graph properties across different strategies.

A `RunReporter` instance is automatically returned by both `Enhancer.compare_strategies()` and `Enhancer.process_tasks()`.

### `get_comparison(predict_metrics, graph_metrics)`

Evaluates each run using a combination of predictive and structural metrics.

**Parameters**:

- `predict_metrics` (optional): A list of predictive metric functions. Each function should accept a pair of `NumPy` arrays representing the true and predicted values and return a single numeric score (`float`).
- `graph_metrics` (optional): A list of functions for computing descriptive graph properties. Each function should take a [`networkx.Graph`](https://networkx.org/documentation/stable/reference/classes/graph.html) object as input and return a numerical value (`float`) that characterizes some structural aspect of the graph.

If no custom metrics are provided, a set of default graph metrics is computed automatically: graph density, average node degree, number of connected components, and size of the largest component.

**Returns**:

- A dictionary mapping each run identifier to a dictionary of computed metric values.

### `get_graphs()`

Returns the graph representation constructed from the edge index for each run.

**Returns**:

- An iterator over tuples `(name, graph)`, where `name` is the run identifier and `graph` is a [`networkx.Graph`](https://networkx.org/documentation/stable/reference/classes/graph.html) instance reconstructed from the strategy output.

This can be used for additional structural analysis or visualization.

### `plot_train_logs(save_to=None)`

Generates training curves for all recorded runs, visualizing the training and validation loss across epochs. Each subplot corresponds to a strategy execution.

**Parameters**:

- `save_to` (optional): A Python `Path` object. If provided, the figure will be saved to the specified location. If omitted, the plot is displayed interactively.
