# Binomial Logistic Regression: Activity Recognition Study
## Overview
This project demonstrates how to construct and evaluate a **binomial logistic regression model** in Python using `scikit-learn`.  
The dataset, **activity.csv**, comes from a motion detection study on elderly participants and was adapted from the **UCI Machine Learning Repository**.  
The goal is to predict whether a participant is **lying down (1)** or **not lying down (0)** based on their **vertical acceleration (Acc vertical)**.
## Dataset
**Source:** Modified dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).  
- **Rows:** 494 observations  
- **Columns:** 2 variables
## Dataset
The dataset has two columns: 

| **Column Name**  | **Type**  | **Description** |
|------------------|-----------|-----------------|
| Acc (vertical)   | float64   | Represents the acceleration of a participant’s vertical movement while performing activities. |
| LyingDown        | int64     | Binary target variable. 1 = participant lying down, 0 = performing another activity. |
## Steps in Analysis
1-**Import Libraries and Load Data**
Let us import the relevant libraries and load the dataset. 

```
import pandas as pd
import seaborn as sns
activity = pd.read_csv('/content/activity.csv')
activity.head()
```
<img width="265" height="200" alt="image" src="https://github.com/user-attachments/assets/acf42458-9282-4f49-933f-ad59ab512b34" />

Let us check how many rows and column the datset has. 

```
activity.shape
```
(494, 2)


Undertanding the datatypes in eah column

```
activity.dtypes
```

<img width="175" height="83" alt="image" src="https://github.com/user-attachments/assets/41cc33b1-91fd-4feb-800d-1e0bf7f8678b" />
 Let's use the descripe function to know about statstics.

```
activity.describe()

```
<img width="293" height="285" alt="image" src="https://github.com/user-attachments/assets/2714ddb6-6d8f-4fcb-97d3-b12b8476a4e1" />


We can see from the table that no data is missing. LyingDown is categorical variable and Acc(Vertical) is numeric, whose mean is 45.51, median 41.10, and it ranges from -48.45 till 112.31. 

2. **Data Preparation**

Now we preparing data for the model.

```
x = activity[['Acc (vertical)']]
y = activity[['LyingDown']]
```
Importing the relevant libraries for the model

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```
Splitting the data into train and test and build the model.


```
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=42)
clf = LogisticRegression().fit(X_train, y_train)
```
3. **Coefficient and Visualising the Model**
The coefficient is

```
clf.coef_
```
array([[-0.1177471]])
y-intercept

```
clf.intercept_
```
array([6.10180958])

Lets plot a regplot graph 
```
ns.regplot(x=activity['Acc (vertical)'], y=activity['LyingDown'], logistic=True)
```
<img width="604" height="447" alt="image" src="https://github.com/user-attachments/assets/dc35fa3f-303d-4e7c-8188-6c2d29420c4d" />

It can be shown from the graph as vertical activity increase the probability of lying down decreases. 

4. **Model Interpretation**
*Technical Interpretation*
- Coefficient (slope) = -0.1177
  - Represents the log-odds change of lying down for a one-unit increase in vertical acceleration.
  - In odds terms e^(-0.1177) ≈ 0.889.  Each 1-unit increase in acceleration decreases the odds of lying down by about 11%.
-Intercept = 6.1018
  - Represents the log-odds of lying down when acceleration = 0.
  - Probability = e^(6.1018) / (1 + e^(6.1018)) ≈ 0.9975, nearly 100% probability of lying down when there’s no vertical movement.

Higher vertical movement means the person is less likely to be lying down.If vertical movement is near zero, they are almost certainly lying down.The model can therefore predict lying-down behaviour from sensor data alone.

5. **Model Evaluation**
Lets test the modell on holdout sample.
```
import sklearn.metrics as metrics
import numpy as np
y_pred = clf.predict(x_test)
clf.predict(x_test)
```
<img width="642" height="130" alt="image" src="https://github.com/user-attachments/assets/6ce40b50-48c0-4425-97ed-cb6aef7fccd0" />

Lets check the probability. 
```
probs = clf.predict_proba(x_test)[:,-1]
np.set_printoptions(suppress=True, precision=4)  # 4 decimal places
print(probs)
```
<img width="617" height="271" alt="image" src="https://github.com/user-attachments/assets/053cc21c-96b0-479e-a076-5917838cf31a" />

Now we create a confusion matrix. 
```
cm = metrics.confusion_matrix(y_test, y_pred, labels= clf.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= clf.classes_)
disp.plot()
```
<img width="710" height="452" alt="image" src="https://github.com/user-attachments/assets/02eda2e4-7fb9-488c-9d18-12d0cd66c82e" />

To interpret the numbers in the graph.
- The upper-left quadrant displays the number of true negatives.
- The bottom-left quadrant displays the number of false negatives.
- The upper-right quadrant displays the number of false positives.
- The bottom-right quadrant displays the number of true positives.

In our model
- True negatives: The number of people that were not lying down that the model accurately predicted were not lying down.
- False negatives: The number of people that were lying down that the model inaccurately predicted were not lying down.
- False positives: The number of people that were not lying down that the model inaccurately predicted were lying down.
- True positives: The number of people that were lying down that the model accurately predicted were lying down.

It is now time to calcuate accuracy, precision, and recall metrics 
```
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
roc_auc = metrics.roc_auc_score(y_test, y_pred)
# Printing them
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)
```
<img width="258" height="80" alt="image" src="https://github.com/user-attachments/assets/28f2ed63-3265-4728-a070-ea0ef4c848fa" />

Let's do the final evaluation and draw ROC Curve:

**ROC Curve**:

```
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_pred)
plt.show()
```
<img width="505" height="438" alt="image" src="https://github.com/user-attachments/assets/f6350ff7-297e-42a8-b34a-1c36d533a623" />

## Result

- The logistic regression model shows high precision, recall, and accuracy, demonstrating excellent predictive power.
- Even simple motion sensor data can accurately detect lying-down behaviour in elderly subjects.



