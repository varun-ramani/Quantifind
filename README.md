# Quantifind

## Overview
Quantifind is a project aimed at finding optimal quantization strategies for diverse neural networks. The project includes a custom quantization framework and a suite of neural network models to evaluate the effects of different quantization levels.

## Source Documentation
The `src` directory has the following structure:
- data: Contains modules `language_names`, `mnist_data`, and
  `simple_dense_data`. Each of these modules has a function
  `create_dataloaders()`, which when invoked with no arguments, returns the
  `train`, `val`, and `test` dataloaders (in that order) for the respective data
  modeled within each module.
  - `language_names`: Data maps names to countries of origin. Refer to
    `datasets/languagenames.tar`, which contains a series of files of the form
    `<Nationality>.txt`, each containing a series of names `<n1> <n2> <n3>`. Therefore, this dataset explodes out the names into key value pairs of the form `<n1>: <Nationality>`. Of course, names are converted to numerical, uniform-length tensors and nationalities are converted to numbers denoting classes
  - `mnist_data`: Loads in the MNIST digits dataset from Torchvision. No
    transforms applied; just converts the images to tensors before returning
    them.
  - `simple_dense_data`: Computes a randomly chosen (with seed, so it doesn't
    change between executions) linear transformation $\mathbb{F}:
    \mathbb{R}^{256} \rightarrow \mathbb{R}^{256}$. Randomly (with seed, again)
    chooses large set of inputs and maps them to outputs with the function. 
- models: Contains the neural network implementations `conv_mnist_model`,
  `dense_mnist_model`, `language_transformer_model`, and `simple_dense_model`.
  Each module contains a `create_model_context()` function that, when called
  without parameters, returns an instance of the model, a criterion (loss
  function), and an optimizer. 
- utils: Some utility code for logging and managing globals. Nothing much to see here.
- workflow: Contains the `data_workflow`, `evaluation_workflow`,
  `quantize_workflow`, `train_workflow`, and `visualization_workflow` modules. 
  - `data_workflow`: Contains the concrete implementation of the
    `create_dataloaders()` function, but you really don't need to interact with
    this.
  - `evaluation_workflow`: Contains the implementations of the evaluation methods described in the paper. 
    - `compute_test_loss`: Lowest level function; given model, test loader, and
      criterion, this computes the criterion for the model on the test loader's data.
    - `evaluate_quantized_model`: Given a model, criterion, test loader, and bit
      depth, this quantizes the model and evaluates it on the loader at the
      quantized depth.
    - `evaluate_bit_depths`: Given a model, criterion, and test loader, computes
      `evaluate_quantized_model` for all the possible bit depths.
  - `quantize_workflow`: Contains the `BitQuantizer(bit_depth)` class. You really only need
    to know about the following two methods:
    - `quantize`: Accepts a tensor and returns the quantized (discretized) variant.
    - `quantize_model`: Accepts an entire model, then deepcopies it and
      quantizes all the weights of the copy. Returns the copy (so original
      remains intact)
    - `quantized_dataloader`: Accepts a dataloader and wraps it with a quantizer
      to quantize the inputs.
  - `train_workflow`: Supplies the core train_network function and some
    load/save utilities. I'll elaborate on the training function below.
    - `save_train_context`: Accepts a checkpoints directory (creating it if it
      doesn't exist), a model, and an optimizer. Saves the checkpoint.
    - `load_train_context`: Accepts a checkpoint directory, a model, and an
      optimizer. Loads the most recent checkpoint from the directory into the
      model and optimizer.
  - `visualization_workflow`: Contains functionality to draw graphs.
    - `visualize_bit_depth_results`: Accepts the result of `evaluate_bit_depths`
      and generates graphs for all the metrics described. 

## Getting Started
1. Create a Conda environment from the `environment.yml` file provided in this repository.
2. Create a Jupyter notebook at the root level (i.e. above src)
3. Run the following cell in the Jupyter notebook:
   ```py
   import sys
   sys.path.append('src')
   ```
4. This is a "universal block". I designed the framework to be hot swappable; to
   change to a different model, just change the data source and model. I have
   provided the blocks for all four models, but it's worth taking a second to
   notice that they're all extremely similar.
   -  ```py
      from data import simple_dense_data   
      from models import simple_dense_model
      from workflow.train_workflow import train_network
      from workflow.evaluation_workflow import evaluate_bit_depths
      from workflow.visualization_workflow import visualize_bit_depth_results

      train_loader, val_loader, test_loader = simple_dense_data.create_dataloaders()
      net, crit, optim = simple_dense_model.create_model_context()
      train_network(net, crit, optim, train_loader, val_loader, epochs=8)
      ```
   -  ```py
      from data import mnist_data   
      from models import conv_mnist_model
      from workflow.train_workflow import train_network
      from workflow.evaluation_workflow import evaluate_bit_depths
      from workflow.visualization_workflow import visualize_bit_depth_results

      train_loader, val_loader, test_loader = mnist_data.create_dataloaders()
      net, crit, optim = conv_mnist_model.create_model_context()
      train_network(net, crit, optim, train_loader, val_loader, epochs=8)
      visualize_bit_depth_results(evaluate_bit_depths(net, test_loader, crit))
      ```
   -  ```py
      from data import mnist_data   
      from models import dense_mnist_model
      from workflow.train_workflow import train_network
      from workflow.evaluation_workflow import evaluate_bit_depths
      from workflow.visualization_workflow import visualize_bit_depth_results

      train_loader, val_loader, test_loader = mnist_data.create_dataloaders()
      net, crit, optim = dense_mnist_model.create_model_context()
      train_network(net, crit, optim, train_loader, val_loader, epochs=8)
      visualize_bit_depth_results(evaluate_bit_depths(net, test_loader, crit))
      ```
   -  ```py
      from data import language_names   
      from models import language_transformer_model
      from workflow.train_workflow import train_network
      from workflow.evaluation_workflow import evaluate_bit_depths
      from workflow.visualization_workflow import visualize_bit_depth_results

      train_loader, val_loader, test_loader = language_names.create_dataloaders()
      net, crit, optim = language_transformer_model.create_model_context()
      train_network(net, crit, optim, train_loader, val_loader, epochs=8)
      visualize_bit_depth_results(evaluate_bit_depths(net, test_loader, crit))
      ```