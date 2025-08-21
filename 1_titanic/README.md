# Titanic Survival Prediction 

This project predicts the survival during the titanic disaster based on socio-economic passengers data. The projects features data cleaning, feature engineering, one-hot encoding, feature selection and classifier fitting. The best classifier is Random Forest, with a train accuracy of 0.98 and an F1-score of 0.98. The kaggle submission scores 0.77 on the test set.

## Install

Install Anaconda or Miniconda, install and activate the `py-data` environment.

```
conda env create -f environment.yml 
conda activate py-data
```

## Run the notebook

```
jupyter lab
```

## Data Set

The training set is used to build the machine learning models.
The test is used to see how well your model performs on unseen data.


| Variable |                 Definition                 |                       Key                      |
|:--------:|:------------------------------------------:|:----------------------------------------------:|
| survival | Survival                                   | 0 = No, 1 = Yes                                |
| pclass   | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| sex      | Sex                                        |                                                |
| Age      | Age in years                               |                                                |
| sibsp    | # of siblings / spouses aboard the Titanic |                                                |
| parch    | # of parents / children aboard the Titanic |                                                |
| ticket   | Ticket number                              |                                                |
| fare     | Passenger fare                             |                                                |
| cabin    | Cabin number                               |                                                |
| embarked | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |


- *pclass*: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

- *age*: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

- *sibsp*: The dataset defines family relations in this way :
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

- *parch*: The dataset defines family relations in this way :
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.


