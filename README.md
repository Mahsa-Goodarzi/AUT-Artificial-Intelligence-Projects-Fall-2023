# AI Course Projects - Fall 2023

This repository contains a collection of projects developed for the **Artificial Intelligence** course at **Amirkabir University of Technology (Tehran Polytechnic)** during the Fall 2023 semester. The projects explore various fundamental AI concepts ranging from adversarial search to machine learning.

## Projects Overview

### 1. Pac-Man Agent using Minimax Algorithm 
**Goal:** Implement an intelligent agent capable of playing a simplified version of Pac-Man.
* **Description:** The project involves designing an agent that uses the **Minimax Algorithm** to make optimal decisions to eat dots while avoiding ghosts. The focus is on defining an effective utility function and exploring the game tree.
* **Key Concepts:** Game Theory, Minimax Algorithm, Utility Functions, Adversarial Search.
* **Tech Stack:** Python.

### 2. Kakuro Puzzle Solver (CSP) 
**Goal:** Solve the Kakuro logic puzzle using Constraint Satisfaction Problem (CSP) techniques.
* **Description:** This project implements two different agents to solve the puzzle:
    1.  **Simple Agent:** Uses a standard **Backtracking** algorithm.
    2.  **Smart Agent:** Enhances backtracking with CSP heuristics (such as Forward Checking or MRV) to prune the search space and improve performance.
* **Key Concepts:** Constraint Satisfaction Problems (CSP), Backtracking, Heuristics, Pruning.
* **Tech Stack:** Python (Itertools).

### 3. Credit Card Fraud Detection 
**Goal:** Detect fraudulent credit card transactions using Machine Learning.
* **Description:** A comparative study using both unsupervised and supervised learning methods to identify fraud in a highly imbalanced dataset.
    * **Unsupervised:** Used **K-Means** clustering.
    * **Supervised:** Used **Logistic Regression** for classification.
* **Key Concepts:** Supervised Learning, Unsupervised Learning, Anomaly Detection, Classification Metrics (Precision, Recall, F1-Score).
* **Tech Stack:** Python, Pandas, Scikit-learn, Matplotlib.

### 4. Iris Dataset Clustering (Genetic Algorithms) 
**Goal:** Cluster the Iris flower dataset using Evolutionary Algorithms.
* **Description:** This project implements a **Genetic Algorithm** to perform clustering on the Iris dataset. The results (Accuracy/Purity) are compared against the standard **K-Means** clustering algorithm to evaluate the evolutionary approach's effectiveness.
* **Key Concepts:** Genetic Algorithms, Evolutionary Computing, K-Means Clustering, Unsupervised Learning.
* **Tech Stack:** Python, NumPy, Scikit-learn.

## General Requirements
To run these notebooks, you will generally need:
* Python 3.x
* Jupyter Notebook / Google Colab
* Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

## Author
**Mahsa Goodarzi** Amirkabir University of Technology  
Fall 2023
