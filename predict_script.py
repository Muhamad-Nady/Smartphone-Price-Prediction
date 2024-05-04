import joblib
import pandas as pd

# Load the saved XGBoost model
loaded_model = joblib.load('best_xgb_model.pkl')

# Prepare input data for prediction (example data)
# You need to prepare your input data accordingly
input_data = pd.DataFrame({
    'battery_power': [2000],
    'blue': [1],
    'clock_speed': [2.0],
    'dual_sim': [1],
    'fc': [5],
    'four_g': [1],
    'int_memory': [32],
    'm_dep': [0.5],
    'mobile_wt': [150],
    'n_cores': [4],
    'pc': [10],
    'px_height': [1000],
    'px_width': [800],
    'ram': [4000],
    'sc_h': [12],
    'sc_w': [5],
    'talk_time': [10],
    'three_g': [1],
    'touch_screen': [1],
    'wifi': [1]
})

# Make predictions using the loaded model
predictions = loaded_model.predict(input_data)

# Display the predictions
print("Predicted Price Range:", predictions)
