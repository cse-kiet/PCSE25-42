from django.shortcuts import render, redirect
from django.contrib.auth.models import User 
# Create your views here.
from django.contrib import messages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from . models import *
import pickle
from tensorflow.keras import layers, models

import requests
from django.shortcuts import render

def userhome(request):
    temperature = None
    error = None
    
    if request.method == "POST":
        location = request.POST.get("location")
        latitude = request.POST.get("latitude")
        longitude = request.POST.get("longitude")

        # Hardcoded values for testing
        if not latitude or not longitude:
            latitude = 28.6523392
            longitude = 77.2636672

        print(f"Latitude: {latitude}, Longitude: {longitude}")

        # Open-Meteo API URL with formatted parameters
        base_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m"

        response = requests.get(base_url)
        data = response.json()
        print(data)
        if response.status_code == 200:
                temperature = data["current"]["temperature_2m"]
        else:
                error = data.get("message", "Error fetching data")
                
    return render(request, "userhome.html", {"temperature": temperature, "error": error})


def index(request):

    return render(request,'index.html')

def about(request):
    
    return render(request,'about.html')

def login(request):
    if request.method=='POST':
        lemail=request.POST['email']
        lpassword=request.POST['password']

        d=Register.objects.filter(email=lemail,password=lpassword).exists()
        print(d)
        return redirect('userhome')
    else:
        return render(request,'login.html')

def registration(request):
    if request.method=='POST':
        Name = request.POST['Name']
        email=request.POST['email']
        password=request.POST['password']
        conpassword=request.POST['conpassword']
        

        print(Name,email,password,conpassword)
        if password==conpassword:
            rdata=Register(email=email,password=password)
            rdata.save()
            return redirect('login')
        else:
            msg='Register failed!!'
            return render(request,'register.html')

    return render(request,'register.html')
    # return render(request,'register.html')


# def userhome(request):
    
#     return render(request,'userhome.html')

def load(request):
   if request.method=="POST":
        file=request.FILES['file']
        global df
        df=pd.read_csv(file)
        messages.info(request,"Data Uploaded Successfully")
    
   return render(request,'load.html')

def view(request):
    # Assuming df is your DataFrame loaded from some source
    # df = pd.read_csv('your_file.csv')  # Example for loading CSV

    # Getting the first 100 rows
    dummy = df.head(100)

    # Extract columns and rows for rendering
    col = dummy.columns.tolist()  # Get column names as a list
    rows = dummy.values.tolist()  # Convert rows to a list of lists

    # Render the template with the data
    return render(request, 'view.html', {'col': col, 'rows': rows})
    
  
def preprocessing(request):

    global x_train,x_test,y_train,y_test,X,y
    if request.method == "POST":
        # size = request.POST['split']
        size = int(request.POST['split'])
        size = size / 100
        df.drop('date',axis=1,inplace=True)
        le = LabelEncoder()
        df['weather'] = le.fit_transform(df['weather'])

        #Preprocess Data for Machine Learning Development
        X = df.drop(['weather'], axis = 1)
        y = df['weather']

        oversample = SMOTE(random_state=1)
        X_final, Y_final = oversample.fit_resample(X, y)

        x_train, x_test, y_train, y_test = train_test_split(X_final, Y_final, test_size = 0.2, random_state = 10)
        x_train.shape, x_test.shape, y_train.shape, y_test.shape

        messages.info(request,"Data Preprocessed and It Splits Succesfully") 
    return render(request,'preprocessing.html')
 

def model(request):
    if request.method == "POST":

        model = request.POST['algo']

        if model == "1":
            lr = LogisticRegression(max_iter=2000)
            lr.fit(x_train,y_train)
            y_pred = lr.predict(x_test)
            acc_lr=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of LogisticRegression :  ' + str(acc_lr)
            return render(request,'model.html',{'msg':msg})

        elif model == "2":
            with open('CNN_model.pkl', 'rb') as fp:
                mod=pickle.load(fp)
            y_pred = mod.predict(x_test)
            y_pred_classes = y_pred.argmax(axis=-1)  # Get the index of the highest probability
            accuracy = accuracy_score(y_test, y_pred_classes)
            print(f"Accuracy: {accuracy}")
            msg = 'Accuracy of CNN :  ' + str(accuracy)
            return render(request,'model.html',{'msg':msg})      
        
        elif model == "3":
            mlp=MLPClassifier(max_iter=2000)
            mlp.fit(x_train,y_train)
            y_pred = mlp.predict(x_test)
            acc_mlp=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of MLP :  ' + str(acc_mlp)
            return render(request,'model.html',{'msg':msg}) 
        
        elif model == "4":
            rf_hyp = RandomForestClassifier(max_depth = None, max_features = 'log2', min_samples_leaf = 1, min_samples_split = 2, n_estimators = 200)
            rf_hyp.fit(x_train, y_train)
            y_pred = rf_hyp.predict(x_test)
            acc_rf=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of Random Forest :  ' + str(acc_rf)
            return render(request,'model.html',{'msg':msg})  
        
        elif model == "5":
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            y_pred = dt.predict(x_test)
            acc_dt=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of Decision Tree :  ' + str(acc_dt)
            return render(request,'model.html',{'msg':msg})   
    return render(request,'model.html')


def prediction(request):
    if request.method == 'POST':

        f1 = float(request.POST['precipitation'])
        f2 = float(request.POST['temp_max'])
        f3 = float(request.POST['temp_min'])
        f4 = float(request.POST['wind'])

        PRED = [[f1,f2,f3,f4]]
       
        rf_hyp = RandomForestClassifier(max_depth = None, max_features = 'log2', min_samples_leaf = 1, min_samples_split = 2, n_estimators = 200)
        rf_hyp.fit(x_train, y_train)
        RES = rf_hyp.predict(PRED)
        RES=RES[0]
         
        if RES==0:
            msg = ' <span style = color:white;>The Weather is going to be : <span style = color:green;><b>Drizzle</b></span></span>'
        elif RES==1:
            msg = ' <span style = color:white;>The Weather is going to be: <span style = color:red;><b>Fog</b></span></span>'
        elif RES==2:
            msg = ' <span style = color:white;>The Weather is going to be: <span style = color:red;><b>Rain</b></span></span>'
        elif RES==3:
            msg = ' <span style = color:white;>The Weather is going to be: <span style = color:red;><b>Snow</b></span></span>'
        else :
            msg = ' <span style = color:white;>The Weather is going to be: <span style = color:red;><b>Sun</b></span></span>'
        
        return render(request,'prediction.html',{'msg':msg})

    return render(request,'prediction.html')