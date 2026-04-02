# Task 1 — Reproduce and Identify Leakage
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(
    X_scaled,y,test_size=0.25,random_state=42
)
model=LogisticRegression()
model.fit(X_train,y_train)
print("Accuracy:",model.score(X_test,y_test))


# Task 2 — Fix the Workflow Using a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X,y=make_classification(n_samples=1000,n_features=10,random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
pipe=Pipeline([
    ('scaler',StandardScaler()),
    ('model',LogisticRegression())
])


scores=cross_val_score(pipe,X,y,cv=5,scoring="accuracy")
print("Mean accuracy:",scores.mean().round(2))
print("Std deviation:",scores.std().round(2))

# Task 3 — Experiment with Decision Tree Depth
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

result=[]
X,y=make_classification(n_samples=200,n_features=5,random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
for depth in [1,5,20]:
  model=DecisionTreeClassifier(max_depth=depth,random_state=42)
  model.fit(X_train,y_train)
  train_acc=model.score(X_train,y_train)
  test_acc=model.score(X_test,y_test)
  result.append([depth,round(train_acc,2),round(test_acc,2)])
print("Depth | Train accuracy | Test accuracy")
for row in result:
  print(f"{row[0]:5}|{row[1]:14}|{row[2]:14}")