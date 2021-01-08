![GMIT Logo](http://password.gmit.ie/images/logo.png "GMIT Logos")
# Machine Learning and Statistics

Repository for the project for the Machine Learning and Statistics module @ GMIT - 2020

Author: Maciej Izydorek Email: G00387873@gmit.ie Github: [mizydorek](https://github.com/mizydorek)

#

#### Project Overview

Create a web service that uses machine learning to make predictions based on the powerproduction data set. The goal is to produce a model that accurately predicts wind turbine power output from wind speed values, as in the data set. You must then develop a web service that will respond with predicted power values based on speed values sent as HTTP requests.

#### Contents of repository

The repository contains:

* `README.md` file
* `project.ipynb` — jupyter notebook contains solution to the project
* `app.py` — python script that contains application based on flask package 
* `Dockerfile` — docker file
* `static` — folder contains index.html page for the application
* `requirements.txt` — contains list of packages need to run the notebook
* `.gitignore` — contains list of files that have to be ignore
* `.dockerignore` — contains list of files that have to be ignore
* `assessment.pdf` — contains project overview and requirements
* `powerproduction.csv` — Power production dataset

#### Required software

The following software is required to run the notebook:

* `git` — open source software to download the repository onto computer[3]
* `python` — Python Programming Language
* `Numpy. Matplotlib. Seaborn. Jupyter. Pandas. SciPy. Sklearn. Flask` — python packages used to make an investigation into the numpy.random package[5,6,7,8,9,10] 

The following command will install the packages according to the configuration file

```
$ pip install -r requirements.txt
```

or alternatively 

* `Anaconda` — Individual edition of Anaconda can be install to get all of the relevant software.[4]

#### Instructions for downloading repository

At github repository [Machine Learning and Statistics](https://github.com/mizydorek/Machine-Learning-Project-2020) click on the green `Code` button to copy the link. Open command line(windows) or terminal(osx/linux), navigate to the selected directory and enter the command `git clone` followed copied the URL. Alternatively copy the whole command below:

```
git clone https://github.com/mizydorek/Machine-Learning-Project-2020.git
```

The whole repository will be cloned down onto current working directory.

#### How to run the jupyter notebook

From there run `jupyter notebook` command on the command line/terminal. This will open Jupyter in the browser. The notebook containing solution that is called `project.ipynb` can be opened by clicking on it.
Once opened, select `Restart and Run All` from Kernel sub-menu to run the jupyter notebook.

#### Viewing the Notebook 

The notebook can be viewed online either on the [github](https://github.com/mizydorek/Machine-Learning-Project-2020/blob/main/project.ipynb) or by accessing through Jupyter Notebooks viewer  [nbviewer](https://nbviewer.jupyter.org/github/mizydorek/Machine-Learning-Project-2020/blob/main/project.ipynb).

#### How to run the application

From current working directory start the web app using the command in terminal **osx/linux** 

```
export FLASK_APP=app.py
```

or command line in **windows**

```
set FLASK_APP=app.py
```

To run the server

```
python -m flask run
```

Localhost server can be accessed at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

To stop server running press `ctrl` + `c` in terminal.  


Alternatively using Docker 

Install Docker using the command below 

```
pip install docker
```

In current working directory run the following command to build a docker image 

```
docker build . -t wind-turbine-app
```

In order to start docker container enter the command 

```
docker run -i -t -p 5000:5000 --rm wind-turbine-app
```

Localhost server can be accessed at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

To stop server running press `ctrl` + `c` in terminal.  

#### References 

[1] Python (https://www.python.org/downloads/) 

[2] Project Jupyter. Jupyter notebook. (http://jupyter.org/)

[3] Software Freedom Conservancy. Git. (https://git-scm.com/)

[4] Anaconda. Individual Edition. (https://www.anaconda.com/products/individual)

[5] NumPy developers. Numpy. (http://www.numpy.org/)

[6] Matplotlib (https://matplotlib.org/)

[7] Seaborn (https://seaborn.pydata.org/)

[8] Pandas (https://pandas.pydata.org/)

[9] Scipy (https://www.scipy.org/)

[10] Sklearn (https://scikit-learn.org/stable/)

[11] Flask (https://flask.palletsprojects.com/en/1.1.x/)

[12] Docker (https://docker.com/)

[13] GMIT. Quality assurance framework. (https://www.gmit.ie/general/quality-assurance-framework)

