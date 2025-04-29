from django.shortcuts import render
from .ml_models import predict_heart_disease, predict_lung_disease, predict_diabetes, predict_liver_disease

def home(request):
    return render(request, 'home.html')

def heart_prediction(request):
    prediction_result = None
    if request.method == 'POST':
        # Get form data
        age = int(request.POST.get('age'))
        sex = int(request.POST.get('sex'))
        cp = int(request.POST.get('cp'))
        trestbps = int(request.POST.get('trestbps'))
        chol = int(request.POST.get('chol'))
        fbs = int(request.POST.get('fbs'))
        restecg = int(request.POST.get('restecg'))
        thalach = int(request.POST.get('thalach'))
        exang = int(request.POST.get('exang'))
        oldpeak = float(request.POST.get('oldpeak'))
        slope = int(request.POST.get('slope'))
        ca = int(request.POST.get('ca'))
        thal = int(request.POST.get('thal'))
        
        # Make prediction
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        prediction_result = predict_heart_disease(features)
        
    return render(request, 'heart.html', {'prediction_result': prediction_result})

def lung_prediction(request):
    prediction_result = None
    if request.method == 'POST':
        # Get form data
        age = int(request.POST.get('age'))
        smoking = int(request.POST.get('smoking'))
        yellow_fingers = int(request.POST.get('yellow_fingers'))
        anxiety = int(request.POST.get('anxiety'))
        peer_pressure = int(request.POST.get('peer_pressure'))
        chronic_disease = int(request.POST.get('chronic_disease'))
        fatigue = int(request.POST.get('fatigue'))
        allergy = int(request.POST.get('allergy'))
        wheezing = int(request.POST.get('wheezing'))
        alcohol = int(request.POST.get('alcohol'))
        coughing = int(request.POST.get('coughing'))
        shortness_of_breath = int(request.POST.get('shortness_of_breath'))
        swallowing_difficulty = int(request.POST.get('swallowing_difficulty'))
        chest_pain = int(request.POST.get('chest_pain'))
        
        # Make prediction
        features = [age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease,
                   fatigue, allergy, wheezing, alcohol, coughing, shortness_of_breath,
                   swallowing_difficulty, chest_pain]
        prediction_result = predict_lung_disease(features)
        
    return render(request, 'lung.html', {'prediction_result': prediction_result})

def diabetes_prediction(request):
    prediction_result = None
    if request.method == 'POST':
        # Get form data
        age = int(request.POST.get('age'))
        glucose = int(request.POST.get('glucose'))
        bmi = float(request.POST.get('bmi'))
        blood_pressure = int(request.POST.get('blood_pressure'))
        insulin = int(request.POST.get('insulin'))
        diabetes_pedigree = float(request.POST.get('diabetes_pedigree'))
        pregnancies = int(request.POST.get('pregnancies'))
        skin_thickness = int(request.POST.get('skin_thickness'))
        
        # Make prediction
        features = [age, glucose, bmi, blood_pressure, insulin, diabetes_pedigree,
                   pregnancies, skin_thickness]
        prediction_result = predict_diabetes(features)
        
    return render(request, 'diabetes.html', {'prediction_result': prediction_result})

def liver_prediction(request):
    prediction_result = None
    if request.method == 'POST':
        # Get form data
        age = int(request.POST.get('age'))
        gender = int(request.POST.get('gender'))
        total_bilirubin = float(request.POST.get('total_bilirubin'))
        direct_bilirubin = float(request.POST.get('direct_bilirubin'))
        alkaline_phosphotase = int(request.POST.get('alkaline_phosphotase'))
        alamine_aminotransferase = int(request.POST.get('alamine_aminotransferase'))
        aspartate_aminotransferase = int(request.POST.get('aspartate_aminotransferase'))
        total_proteins = float(request.POST.get('total_proteins'))
        albumin = float(request.POST.get('albumin'))
        albumin_globulin_ratio = float(request.POST.get('albumin_globulin_ratio'))
        
        # Make prediction
        features = [age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase, 
                   alamine_aminotransferase, aspartate_aminotransferase, 
                   total_proteins, albumin, albumin_globulin_ratio]
        prediction_result = predict_liver_disease(features)
        
    return render(request, 'liver.html', {'prediction_result': prediction_result})