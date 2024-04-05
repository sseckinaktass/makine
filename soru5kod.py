import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_excel("normalized_veriler.xlsx") 


X = df.drop(columns=['Class variable (0 or 1)'])  
y = df['Class variable (0 or 1)']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = GaussianNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)


print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


print("Karışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))


