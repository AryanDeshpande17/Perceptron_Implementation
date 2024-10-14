# Perceptron and One-vs-All Perceptron Implementations

This repository contains implementations of the Perceptron algorithm and a One-vs-All (OvA) Perceptron for multi-class classification using the Iris dataset. The scripts demonstrate basic linear classification and how to extend the Perceptron model to handle multiple classes versus only binary classification.
The basic idea for this implementation comes from the research paper, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" ~ F. ROSENBLATT

## Contents

- **perceptron.py**: Implementation of the Perceptron algorithm with visualization of the decision boundary for binary classification of Iris species.
- **multiclass_perceptron.py**: Implementation of the One-vs-All Perceptron algorithm, including PCA, for visualizing the Iris dataset in two dimensions.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn

You can install the required packages using pip:

```bash
pip install numpy matplotlib scikit-learn
```

You can run the Perceptron on the sample Iris Dataset using:

```bash
python perceptron.py
```

And the OnevAll Perceptron using:
```bash
python multiclass_perceptron.py
```
