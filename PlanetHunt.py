import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('cumulative.csv')
df = df.replace({'CONFIRMED': 0, 'CANDIDATE': 1, 'FALSE POSITIVE': 2})

# Clean the Data Frame by removing all labels, and error values
df = df.drop(['rowid', 'kepid', 'kepoi_name', 
        'kepler_name', 'koi_tce_delivname', 
        'koi_period_err1', 'koi_period_err2',
        'koi_time0bk_err1', 'koi_time0bk_err2', 
        'koi_impact_err1', 'koi_impact_err2',
        'koi_depth_err1', 'koi_depth_err2', 
        'koi_prad_err1', 'koi_prad_err2', 
        'koi_insol_err1', 'koi_insol_err2',
        'koi_steff_err1', 'koi_steff_err2', 
        'koi_slogg_err1', 'koi_slogg_err2', 
        'koi_srad_err1', 'koi_srad_err2', 
        'koi_duration_err1', 'koi_duration_err2', 
        'koi_teq_err1', 'koi_teq_err2'], axis=1)

# Dealing with null data fields
df.fillna(df.mean(), inplace=True)
# Split the data into features and labels
X = df.drop('koi_disposition', axis=1)
y = df['koi_disposition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate multiple classification models
models = {
    'Logistic Regression': LogisticRegression(random_state=16),
    'SVC': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
}

accuracy = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy.append(acc)
    print(model)
    print(acc)
    

# # Store the accuracy of each model
# accuracy = []
# for name, model in models.items():
#     scores = cross_val_score(model, X_train, y, cv=5)
#     accuracy.append(scores.mean())
# # Print the average accuracy for each model
# for name, acc in zip(models.keys(), accuracy):
#     print(f"{name}: Average Accuracy = {acc}")


def plot_learning_curve(estimator, title, X, y, cv=None, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, train_sizes=train_sizes)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Accuracy')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Validation Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='g')

    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

#Plot the learning curves for all models
for name, model in models.items():
    plot_learning_curve(model, "Learning Curve", X_train, y_train, cv=5) 