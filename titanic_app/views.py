from django.shortcuts import render
from django.shortcuts import render,HttpResponse,redirect
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def view(request):
    return render(request, 'index.html')

def get_user_input_dia(request):
# Load and preprocess the diabetes dataset
    data2 = pd.read_csv('titanic_train.csv')
    data2=data2.replace(to_replace=["male","female"],value=[1,0])
    data2=data2.dropna(axis=0)
# preprocess the dataset as needed
    columns_drop = ['Survived','Name','Cabin','Embarked','Ticket','PassengerId']
    x = np.array(data2.drop(columns_drop, axis=1))
    y = np.array(data2.Survived)
# Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a logistic regression model
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

# Define a function to take user input

    if request.method == "POST":
        name=request.POST.get('name')
        Pclass=float(request.POST.get('Pclass', 0))
        Sex=float(request.POST.get('Sex', 0))
        Age=float(request.POST.get('Age', 0))
        SibSp=float(request.POST.get('SibSp', 0))
        Parch=float(request.POST.get('Parch', 0))
        Fare=float(request.POST.get('Fare', 0))
        li=[Pclass, Sex, Age, SibSp, Parch, Fare]
        input_array = np.array(li).reshape(1,-1)
        prediction=logreg.predict(input_array)
        
# Call your input function to get user input
    
    if prediction == [1]:
            result = "LUCKY MAN/WOMAN"
            is_affected = False
    else:
            result = "OHH RIP!!!!!!!!! "+name
            is_affected = True

    context = {
            'result' : result,
            'is_affected': is_affected
        }
    return render(request, 'deadoralive.html', context)

# Create your views here.
