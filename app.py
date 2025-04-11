from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Load models
sarima_model = joblib.load('models/sarima_model.joblib')
gradient_model = joblib.load('models/gradient_model.joblib')
kmeans_model = joblib.load('models/kmeans_model.joblib')
rf_model = joblib.load('models/rf_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    model_name = request.form['model']
    file = request.files['csv_file']
    df = pd.read_csv(file)

    # Select model and make predictions
    if model_name == 'sarima':
        prediction = sarima_model.forecast(steps=len(df))
        index = list(range(len(prediction)))
    elif model_name == 'gradient':
        prediction = gradient_model.predict(df)
        index = df.index.tolist()
    elif model_name == 'kmeans':
        prediction = kmeans_model.predict(df)
        index = df.index.tolist()
    elif model_name == 'random_forest':
        prediction = rf_model.predict(df)
        index = df.index.tolist()
    else:
        return "Invalid model selected", 400
    import matplotlib
    matplotlib.use('Agg')

    # Plot the prediction
    plt.figure(figsize=(10, 4))
    plt.plot(index, prediction, marker='o')
    plt.title(f"{model_name.upper()} Model Prediction")
    plt.xlabel("Index")
    plt.ylabel("Prediction")
    plt.grid(True)

    # Save plot to BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
