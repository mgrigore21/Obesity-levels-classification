# Obesity Levels Classification

This project aims to classify obesity levels based on various features using different machine learning models. The project is implemented in Python and uses the Scikit-learn library for model training and evaluation.

## Project Structure

The project consists of the following main files:

- `ml.py`: This file contains the main classification pipeline, which includes data loading, preprocessing, model training, and evaluation.

- `EDA.py`: This file contains exploratory data analysis (EDA) code, which includes functions for reading the data, calculating NaN values, printing most frequent values, and plotting feature distributions and correlation heatmaps.

- `tests.py`: This file contains unit tests for the `ClassificationPipeline` class in `ml.py`.

## How to Run

To run the main classification pipeline, execute the following command:

```bash
python ml.py
```
To run the exploratory data analysis code, execute the following command:

```bash 
python EDA.py
```
To run the unit tests, execute the following command:

```bash
python tests.py
```

Dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
``` 
## Data Source

Estimation of Obesity Levels Based On Eating Habits and Physical Condition . (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5H31Z.

## Results

The project uses the following machine learning models for classification:
```text
Confusion Matrix for Logistic Regression
[[54  0  0  0  0  0  0]
 [ 3 45  0  0  0 10  0]
 [ 0  0 66  1  0  0  3]
 [ 0  0  1 59  0  0  0]
 [ 0  0  0  1 64  0  0]
 [ 0  8  0  0  0 48  2]
 [ 0  0  3  1  0  3 51]]
Precision: 0.9148936170212766
Recall: 0.9148936170212766
F1 Score: 0.9148936170212766


Confusion Matrix for K-Nearest Neighbors
[[52  2  0  0  0  0  0]
 [ 5 40  3  0  0  7  3]
 [ 0  0 68  0  0  1  1]
 [ 0  0  1 59  0  0  0]
 [ 0  0  0  0 65  0  0]
 [ 0  6  1  0  0 49  2]
 [ 0  0  2  0  1  1 54]]
Precision: 0.9148936170212766
Recall: 0.9148936170212766
F1 Score: 0.9148936170212766


Confusion Matrix for Naive Bayes
[[45  2  5  0  0  2  0]
 [24 17  1  0  0 11  5]
 [ 0  1 48 17  0  1  3]
 [ 0  0  5 53  1  1  0]
 [ 0  0  0  0 65  0  0]
 [ 3  6 19  0  2 22  6]
 [ 1  3 30  8  4  1 11]]
Precision: 0.6170212765957447
Recall: 0.6170212765957447
F1 Score: 0.6170212765957447


Confusion Matrix for Random Forest
[[51  3  0  0  0  0  0]
 [ 0 54  0  0  0  3  1]
 [ 0  2 67  0  0  0  1]
 [ 0  0  1 59  0  0  0]
 [ 0  1  0  0 64  0  0]
 [ 0  8  1  0  0 48  1]
 [ 0  1  2  0  0  1 54]]
Precision: 0.9385342789598109
Recall: 0.9385342789598109
F1 Score: 0.9385342789598109


Confusion Matrix for SVM
[[54  0  0  0  0  0  0]
 [ 0 54  0  0  0  4  0]
 [ 0  0 69  1  0  0  0]
 [ 0  0  0 60  0  0  0]
 [ 0  0  0  1 64  0  0]
 [ 0  2  0  0  0 55  1]
 [ 0  0  1  0  0  2 55]]
Precision: 0.9716312056737588
Recall: 0.9716312056737588
F1 Score: 0.9716312056737588


Confusion Matrix for LightGBM
[[51  3  0  0  0  0  0]
 [ 0 57  0  0  0  1  0]
 [ 0  0 68  0  0  2  0]
 [ 0  0  0 60  0  0  0]
 [ 0  0  0  1 64  0  0]
 [ 0  5  0  0  0 52  1]
 [ 0  0  0  0  0  1 57]]
Precision: 0.966903073286052
Recall: 0.966903073286052
F1 Score: 0.966903073286052


Confusion Matrix for Neural Network
[[53  1  0  0  0  0  0]
 [ 2 51  0  0  0  5  0]
 [ 0  0 69  1  0  0  0]
 [ 0  0  0 60  0  0  0]
 [ 0  0  0  0 65  0  0]
 [ 0  6  0  0  0 50  2]
 [ 0  0  0  0  0  1 57]]
Precision: 0.9574468085106383
Recall: 0.9574468085106383
F1 Score: 0.9574468085106383
```
Best result is obtained with the SVM algorithm with a F1 score of 0.9716.