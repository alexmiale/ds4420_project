# ds4420_project

As teams increase revenue and find themselves with lots of data, they look to back up the assessment of analysts and coaches with objective metrics. There are lots of different ways to try to use this data simply (pass success rate), but it is very difficult to quantify how good or bad an action is. 

This project uses machine learning to predict a soccer team's attacking output. We use a neural network to create an expected possession value (EPV) model that predicts the likelihood of a goal being scored from the current possession. We also create an expected goals (xG) model using Bayesian Logistic Regression, predicting the likelihood of any given shot being scored. The data for this project was from SkillCorner, using their 10-game, open data from the Australian A-League. We process the data by combining event and tracking data before sending shots to the xG model and frame-by-frame event data to the EPV model. 
