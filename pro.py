import pandas as p 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


col_names = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Survived']

data = p.read_csv("pro.csv")

features = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Survived']

X = data[features]
y = data.label

x_train , x_test , y_train , y_test  = train_test_split( X , y , test_size = 0.3)

dtc = DecisionTreeClassifier()
dtc = dtc.fit(x_train,y_train)

y_pred = dtc.predict(x_test)

print( metrics.accuracy_score(y_test , y_pred) )

# ------------------------------------- visualizing DT ----------------------------------------

from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image
import pydotplus

# ---------------------- converting data into text -----------------------

dot_data = StringIO()

export_graphviz(dtc , out_file = dot_data , filled=True , rounded = True , special_charachters = True , feature_names = features , class_names = ['0' , '1'])

print(dot_data.getvalue())


# ------------------ converting text into image ------------------------

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('dib.png')
Image(graph.create_png())

# ------------------------------ giving max limit as 3 -----------------------------

dtc = DecisionTreeClassifier(max_depth = 3)
dtc = dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
print( metrics.accuracy_score(y_test , y_pred) )


dot_data = StringIO()

export_graphviz(dtc , out_file = dot_data , filled=True , rounded = True , special_characters = True , feature_names = features , class_names = ['0' , '1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('dib.png')
Image(graph.create_png())
