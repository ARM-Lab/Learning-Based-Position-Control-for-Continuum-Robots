# Learning Based Position Control for Continuum Robots


This repository contains the dataset, CAD files for the testing stand, and the code used to train a Feedforward Neural Network (FFNN) for controlling a continuum robot. The work presented in this repository is part of a research paper on the application of deep learning techniques in controlling the movement of continuum robots.

## Contents

- **Dataset**: Includes data collected during experiments on the continuum robot. This data is used to train the neural network for control.
- **CAD Files**: Provides the 3D CAD models of the testing stand used for experiments. These files are essential for replicating the testing setup and understanding the geometrical constraints.
- **Code**:
  - **Training Code**: Python code for training the Feedforward Neural Network (FFNN) on the dataset.


## Installation

To run the code in this repository, you will need the following dependencies:

1. Python 3.10
2. TensorFlow
3. Keras
4. NumPy
5. Sklearn

### Setting up the Testing Stand (Optional)

If you wish to replicate the testing setup, you can use the provided CAD files to build the testing stand. The CAD files are located in the `TestingBench/` directory. These files are compatible with SolidWorks 2022.

## Dataset

The dataset used for training contains 30 000 action-state pairs. Actions are values between 0 - 100 corresponding to cable shortenings of each of the cables between 0 and 10 mm.  The dataset is stored in txt format in the `Dataset/` directory.

## CAD Files

The CAD files of the testing stand are stored in the `TestingBench/` directory. These files describe the physical layout and configuration of the stand, including the actuators, and robot parts used during the experiment.


## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Citation

If you use this repository or any part of the dataset, CAD files, or code in your work, please cite the following paper:

```
*will be added after succesful review
```
