# import libraries 
import numpy as np 
import flask as fl 

# create a web app. 
app = fl.Flask(__name__)

# Add routes.
@app.route('/')
def hello():
    return app.send_static_file('index.html')

@app.route('/api/normal')
def normal():
    return { 'value': np.random.normal() }