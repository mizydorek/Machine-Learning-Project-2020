# Numerical arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd 

# Import Flask.
import flask as fl 
from flask import request, jsonify, make_response


# Import LinearRegression model from sklearn linear model.
from sklearn.linear_model import LinearRegression
# Using train_test_split() from the scikit-learn library, 
# makes it easy to split dataset into training and testing data.
from sklearn.model_selection import train_test_split

# Import curve_fit from SciPy to Use non-linear least squares to fit a function to data.
from scipy.optimize import curve_fit

# create a web app. 
app = fl.Flask(__name__)

# Add routes.
@app.route('/')
def hello():
    return app.send_static_file('index.html')
'''
@app.route('/api/normal')
def normal():
    return { 'value': np.random.normal() }
'''
@app.route('/api/linear', methods=["POST"])
def model():

    data = request.get_json()

    model = data['model'].lower()
    speed = float(data['speed'])
    #print(req)

    power = eval(model)(speed)
    #power = round(power, 4)

    response = make_response(jsonify(power), 200)

    return response

# Load dataset function.
def load():
    try:
        # Load dataset.
        df = pd.read_csv("https://raw.githubusercontent.com/ianmcloughlin/2020A-machstat-project/master/dataset/powerproduction.csv")
        # Drop columns for power equals 0 above 1m/s wind speed.
        df = df.drop(df[(df.speed > 1) & (df.power == 0)].index).reset_index(drop=True)
        # Split dataframe into speed and power columns using pandas.
        speed = df['speed']
        power = df['power']

        # Convert speed and power columns to numpy arrays.
        x = df.iloc[:, :1].values 
        y = df.iloc[:, 1].values

        return x, y, speed, power
    except:
        print("Failed to load dataset.")

# Preprocessing of dataset
def preprocess():
    try:
        # load data.
        x, y, speed, power = load()
        # Split the dataset into 75% train data and 25% test data.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        return x_train, x_test, y_train, y_test
    except:
        print("Failed to preprocess data.")

# Build linear model.
def linear(speed):
    try:
        # load preprocess data.
        x_train, x_test, y_train, y_test = preprocess()
        # Create a new linear regression model.
        model = LinearRegression()
        # Fit the data to the model.
        model.fit(x_train, y_train)
        # Coefficient & intercept.
        coeff = [model.coef_[0], model.intercept_]
        # Function that count the linear regression.
        return speed * coeff[0] + coeff[1]
    except:
        print("Failed to bulid and train the Linear model.")

# Build polynomial model.
def polynomial(speed):
    try:
        # load data.
        x, y, s, p = load()
        # Define Polynomial degree.
        degree = 9 
        # Fit the data 
        poly = np.poly1d(np.polyfit(s, p, degree))
        #  The polynomial coefficients and intercept.
        coeff = np.poly1d(poly) 
        # Calculate power output.
        power = coeff[9] * speed**9 + coeff[8] * speed**8 + coeff[7] * speed**7 + coeff[6] * speed**6 + coeff[5] * speed**5 + coeff[4] * speed**4 + coeff[3] * speed**3 + coeff[2] * speed**2 +  coeff[1] * speed + coeff[0]

        return power
    except:
        print("Failed to bulid and train the Polynomial model.")

# Logistic function.
def func(speed, a, b, c):
    return a / (1 + b * np.exp(-c * speed))

# Build logistic model. 
def logistic(speed):
    # load data.
    x, y, s, p = load()
    # Initial guess for the parameters.
    bounds=(max(p), np.median(p),min(p))
    # Use non-linear least squares to fit a function to data.
    popt, pcov = curve_fit(func, s, p, bounds)
    return func(speed, *popt)