# Expected Goals (xG) Model with Logistic Regression

This project analyzes soccer shot data to calculate the probability of a shot resulting in a goal (xG).

## 🚀 Project Overview
The model uses historical shot data, including location and match context, to train a machine learning classifier.

## 🛠️ Data Engineering
In this project, I performed several key data transformations:
* **Euclidean Distance**: Calculated the straight-line distance from the shot coordinates $(x, y)$ to the center of the goal $(100, 50)$ using $a^2 + b^2 = c^2$.
* **One-Hot Encoding**: Used `pd.get_dummies()` to convert categorical text like `Zone` (Center, Left, Right, Back) and `period` (FirstHalf, SecondHalf) into numerical binary values.

## 📊 Machine Learning Workflow
1. **Features (X)**: Distance, Shot Angle, and Encoded Zones.
2. **Target (y)**: `is_goal` (Binary).
3. **Model**: Logistic Regression (set to `max_iter=1000` for convergence).
4. **Evaluation**: Assessed using **Log Loss** and **Brier Score** to measure probability accuracy.

## 📦 Requirements
* pandas
* numpy
* scikit-learn
* seaborn
* matplotlib
* mplsoccer
