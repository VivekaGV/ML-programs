import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
values = np.random.rand(100)
labels = []

for i in values[:50]:
    if i <=0.5:
        labels.append('Class1')
    else:
        labels.append('Class2')
labels += [None] * 50
print(labels)

data = {
    "Point": [f"x{i+1}" for i in range(100)],
    "Value": values,
    "Label": labels
}

print(data)
type(data)
df = pd.DataFrame(data)
df.head()
df.nunique()
df.shape
df.info()
df.describe().T
df.isnull().sum()
num_col = df.select_dtypes(include=['int', 'float']).columns
for col in num_col:
    df[col].hist(bins=10, alpha=0.5, edgecolor='black',grid=False)
    plt.title(f'Histogram for {col}')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.show()

labeled_df = df[df["Label"].notna()]
X_train = labeled_df[["Value"]]
y_train = labeled_df["Label"]
unlabeled_df = df[df["Label"].isna()]
X_test = unlabeled_df[["Value"]]
true_labels = ["Class1" if x <= 0.5 else "Class2" for x in values[50:]]

k_values = [1, 2, 3, 4, 5, 20, 30]
results = {}
accuracies = {}
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    results[k] = predictions

    accuracy = accuracy_score(true_labels, predictions) * 100
    accuracies[k] = accuracy
    print(f"Accuracy for k={k}: {accuracy:.2f}%")

    unlabeled_df[f"Label_k{k}"] = predictions

print(predictions)
df1 = unlabeled_df.drop(columns=['Label'], axis=1)
print("\nAccuracies for different k values:")
for k, acc in accuracies.items():
    print(f"k={k}: {acc:.2f}%")