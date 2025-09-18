from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
import time
import psutil
import GPUtil
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from codecarbon import EmissionsTracker
from .recommendation import recommend_models   # ✅ new import


def index(request):
    columns = None
    if request.method == "POST" and request.FILES.get('dataset'):
        dataset_file = request.FILES['dataset']
        df = pd.read_csv(dataset_file)
        columns = list(df.columns)

        # Save uploaded file in session
        request.session['csv_data'] = df.to_json()

    return render(request, 'index.html', {'columns': columns})


def run_model(request):
    if request.method == 'POST':
        selected_features = request.POST.getlist('features')
        target_col = request.POST.get('target')

        if not selected_features or not target_col:
            return HttpResponse("Please select features and target column.")

        df = pd.read_json(request.session['csv_data'])

        X = df[selected_features].values
        y = df[target_col].values

        results = []

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # --- Helper function for resource usage ---
        def measure_resources():
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            gpus = GPUtil.getGPUs()
            gpu_load = gpus[0].load * 100 if gpus else 0
            return cpu, memory, gpu_load

        # --- Helper function for model evaluation ---
        def benchmark_model(model_name, model_obj, X_tr, X_te, y_tr, y_te, is_dl=False):
            tracker = EmissionsTracker()
            tracker.start()

            # record before
            start_cpu, start_mem, start_gpu = measure_resources()
            start = time.time()

            if is_dl:
                model_obj.fit(X_tr, y_tr, epochs=20, batch_size=16, verbose=0)
                y_pred = model_obj.predict(X_te)
            else:
                model_obj.fit(X_tr, y_tr)
                y_pred = model_obj.predict(X_te)

            end = time.time()
            emissions = tracker.stop()

            # record after
            end_cpu, end_mem, end_gpu = measure_resources()

            avg_cpu = round((start_cpu + end_cpu) / 2, 2)
            avg_mem = round((start_mem + end_mem) / 2, 2)
            avg_gpu = round((start_gpu + end_gpu) / 2, 2)

            return {
                'model': model_name,
                'mse': round(mean_squared_error(y_te, y_pred), 4),
                'r2': round(r2_score(y_te, y_pred), 4),
                'runtime': round(end - start, 2),
                'co2_kg': round(emissions, 6),
                'cpu': avg_cpu,
                'ram': avg_mem,
                'gpu': avg_gpu,
            }

        # Traditional ML Models
        results.append(benchmark_model('Linear Regression', LinearRegression(), X_train, X_test, y_train, y_test))
        results.append(benchmark_model('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42),
                                       X_train, X_test, y_train, y_test))
        results.append(benchmark_model('Decision Tree', DecisionTreeRegressor(random_state=42),
                                       X_train, X_test, y_train, y_test))
        results.append(benchmark_model('KNN', KNeighborsRegressor(n_neighbors=5),
                                       X_train, X_test, y_train, y_test))
        results.append(benchmark_model('SVM', SVR(), X_train, X_test, y_train, y_test))

        # LSTM
        X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
        y_lstm = y.reshape(-1, 1)
        split = int(0.8 * len(X_lstm))
        X_train_l, X_test_l = X_lstm[:split], X_lstm[split:]
        y_train_l, y_test_l = y_lstm[:split], y_lstm[split:]

        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_l.shape[1], X_train_l.shape[2])))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer=Adam(), loss='mse')

        results.append(benchmark_model('LSTM', lstm_model, X_train_l, X_test_l, y_train_l, y_test_l, is_dl=True))

        #CNN
        X_cnn = X.reshape((X.shape[0], X.shape[1], 1))
        y_cnn = y.reshape(-1, 1)
        split = int(0.8 * len(X_cnn))
        X_train_c, X_test_c = X_cnn[:split], X_cnn[split:]
        y_train_c, y_test_c = y_cnn[:split], y_cnn[split:]

        cnn_model = Sequential()
        cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_c.shape[1], 1)))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(50, activation='relu'))
        cnn_model.add(Dense(1))
        cnn_model.compile(optimizer=Adam(), loss='mse')

        results.append(benchmark_model('CNN', cnn_model, X_train_c, X_test_c, y_train_c, y_test_c, is_dl=True))
        max_mse = max(r['mse'] for r in results)
        for r in results:
            r['mse_normalized'] = round(r['mse'] / max_mse, 4)
        recommendations = recommend_models(results)

        return render(request, 'index.html', {
            'benchmark_results': results,
            'recommendations': recommendations
        })

    return HttpResponse("Invalid request.")