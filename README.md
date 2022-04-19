# Would I have survived the Titanic?

A classification project with passenger data from the most famous shipwreck. The project is described in [this](https://tomkral.nl/projects/Titanic/titanic.html) article. I have made multiple script that each contain steps for my research. 

### clean_and_EDA.py
This script has two primary tasks:
1. **Clean data:** Raw data is taken in and missing values are deleted. This is also explored with a graph.
2. **Exploratory data analysis:** The data is explored and visualized to gain more insights prior to making a model

This script has several plots and graphs as output. It also makes a new .csv file with the cleaned data.


### fit_model_scaler.py
First, the data is scaled with a min-max-scaler to have all data on the same scale. Second, a suitable _k-value_ is calculated and used to make a k-Nearest Neighbor model. Lastly, both the scaler and model are exported as objects to be loaded later. These can be used in the [webapp](https://github.com/thjkral/titanic_webapp) or in other scripts.