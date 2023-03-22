import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import graphviz
import os

# Sam podjebalem od Chatu GPT, wiec prawa autorskie moje 

# Load data from CSV file
data = pd.read_csv('sport.csv')
# Preprocess data
le = preprocessing.LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col].astype(str))

# Split data into training and testing sets
X = data.drop('klasa', axis=1)
y = data['klasa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build ID3 decision tree
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)

# Evaluate model
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)

# Visualize decision tree
dot_data = export_graphviz(dt, out_file = None, 
                           feature_names=X.columns,  
                           class_names=data['klasa'].astype(str), # astype, bo nazwy klas sa cyframi i wyskakiwal blad bo chcial str
                           filled=True, rounded=True,  
                           special_characters=True)  

# Żeby to gowno dzialalo, trzeba zainstalowac pakiet https://graphviz.org/download po czym skopowiowac tutaj sciezke do pliku bin w tym pakiecie
os.environ["PATH"] += os.pathsep + 'D:/Programy do programowania/Python 3.11/Graphviz/bin'


graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("decision_tree")