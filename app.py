from flask import Flask, render_template, request, jsonify, redirect, url_for
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

app =   Flask(__name__)

# Load the models and scalers
def load_models():
    try:
        lstm_model = tf.keras.models.load_model('models/lstm_vegetable_price_model.h5')
        dense_model = tf.keras.models.load_model('models/dense_vegetable_price_model.h5')
        feature_scaler = joblib.load('models/feature_scaler.pkl')
        target_scaler = joblib.load('models/target_scaler.pkl')
        return lstm_model, dense_model, feature_scaler, target_scaler
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None, None

lstm_model, dense_model, feature_scaler, target_scaler = load_models()

def get_feature_columns():
    return feature_scaler.get_feature_names_out()

def preprocess_input(input_data, feature_scaler):
    try:
        feature_names = get_feature_columns()
        input_dict = {col: 0 for col in feature_names}
        
        # Map numeric features
        numeric_features = ['Temp', 'Rainfall', 'Humidity', 'Demand', 
                          'Freshness', 'Inflation', 'Yield_per_Hectare']
        for feature in numeric_features:
            if feature in input_data:
                input_dict[feature] = float(input_data[feature])
        
        # Map categorical features
        if 'Disaster' in input_data:
            disaster_col = f"Disaster_{input_data['Disaster']}"
            if disaster_col in feature_names:
                input_dict[disaster_col] = 1
        
        if 'Season' in input_data:
            season_col = f"Season_{input_data['Season']}"
            if season_col in feature_names:
                input_dict[season_col] = 1
        
        if 'Holiday_Festival' in input_data:
            holiday_val = 'Yes' if input_data['Holiday_Festival'] != 'No' else 'No'
            holiday_col = f"Holiday_Festival_{holiday_val}"
            if holiday_col in feature_names:
                input_dict[holiday_col] = 1
        
        if 'Vegetable' in input_data:
            veg_col = f"Vegetable_{input_data['Vegetable']}"
            if veg_col in feature_names:
                input_dict[veg_col] = 1
        
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        input_scaled = feature_scaler.transform(input_df)
        
        return input_scaled
    
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {str(e)}")

def make_prediction(input_data):
    try:
        input_scaled = preprocess_input(input_data, feature_scaler)
        lstm_input = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
        
        lstm_prediction_scaled = lstm_model.predict(lstm_input, verbose=0)
        dense_prediction_scaled = dense_model.predict(input_scaled, verbose=0)
        
        lstm_prediction = target_scaler.inverse_transform(lstm_prediction_scaled).flatten()
        dense_prediction = target_scaler.inverse_transform(dense_prediction_scaled).flatten()
        
        ensemble_prediction = (lstm_prediction + dense_prediction) / 2
        
        return float(ensemble_prediction[0])
    
    except Exception as e:
        return {'error': str(e)}

@app.route('/', methods=['GET'])
def home():
    # Explicitly define options for vegetables, disasters, and seasons
    vegetables = [
        "Potato", "Tomato", "Peas", "Pumpkin", "Cucumber", "Ivy Gourd", 
        "Radish", "Bitter Gourd", "Onion", "Garlic", "Cabbage", "Cauliflower"
    ]
    disasters = ["Normal","Cyclone", "Flood", "Drought"]
    seasons = ["Summer", "Winter", "Rainy"]
    
    return render_template('index.html', 
                           vegetables=vegetables,
                           disasters=disasters,
                           seasons=seasons)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'Temp': request.form['temperature'],
            'Rainfall': request.form['rainfall'],
            'Humidity': request.form['humidity'],
            'Demand': request.form['demand'],
            'Freshness': request.form['freshness'],
            'Inflation': request.form['inflation'],
            'Yield_per_Hectare': request.form['yield_per_hectare'],
            'Disaster': request.form['disaster'],
            'Season': request.form['season'],
            'Holiday_Festival': request.form['holiday_festival'],
            'Vegetable': request.form['vegetable']
        }
        
        predicted_price = make_prediction(input_data)
        
        if isinstance(predicted_price, dict) and 'error' in predicted_price:
            return jsonify({'error': predicted_price['error']})
        
        return redirect(url_for('result', 
                              vegetable=input_data['Vegetable'], 
                              price=predicted_price))
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/result')
def result():
    vegetable = request.args.get('vegetable')
    price = request.args.get('price')
    return render_template('result.html', vegetable=vegetable, price=price)

if __name__ == '__main__':
    app.run(debug=True)