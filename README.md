# Linear Regression Model For NBA Games
![Away Plot](https://user-images.githubusercontent.com/73906088/162642442-a6e3ff55-2c6a-478e-b272-2df10bc6e840.png)
![Home Plot](https://user-images.githubusercontent.com/73906088/162642444-1b85cb14-e0de-49b8-8ef7-5c925227dccc.png)
### Summary
This project is used for the purpose of learning data extraction from websites, data cleaning, visualizing data, and creating a linear regression model for an introduction to machine learning. I wanted to see if taking a few key statistics from NBA Games is sufficent to predict the final score of NBA games. I took a few key statistics from all 2022 NBA regular season games: 3P%, FT%, ORB, STL, TOV. I used a dataset of the key statistics and the final score. I split the data into a train and test dataset. I used the key statistics and final scores form the train dataset to make trained model. For the test dataset, I removed the final scores and kept the key statistics. I then used the trained model to predict the final scores of the test dataset. I then compared the actual score for the test dataset with the predicted scores.
### Results
| Statistic        | #  |
|--------------|-----------|
| Mean Squared Error (MSE) | **97.84** |
| Coefficient of Determination (R²) | **0.38**  |

Its suprising how accurate a linear regression model can predict the final score of NBA games. A correlation coefficient of 0.62 is pretty strong for given just five statistics that don't nessiarily relate to direct points for either team. A mean squared error seems high until we take the square root. We only ended up predicting, on average, around 9.892 points off for each actual final score. 

### Libraries Used
- Pandas
- Numpy
- Matplotlib
- Sklearn


























The data was scraped from the http://www.basketball-refrence.com
