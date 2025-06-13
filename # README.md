# README
## Project 4 - NBA Finals Prediction
This project uses random forests to predict the outcome of the NBA Finals from 2019 - 2025. 

## Data
The training data comes from a Kaggle user named Dave Rosenman. I retrieved it via an API in the code and it can also be viewed at this link here: https://www.kaggle.com/datasets/daverosenman/nba-finals-team-stats

For the testing data, I manually created a csv in Excel of the relevant data which I collected from https://www.basketball-reference.com/. I then uploaded it to my own Kaggle account and retrieved it in the code as an API. 

## How to Run
Scroll to the bottom of the code, and you will see the necessary functions commented out of the script using a #. Set the game variable to either game1, game12, or game123, and then remove the # from the functions and run. 

## Notes
I have this code written so that the testing accuracy is a product of the original training data set, and then the model predicts the Finals winners for years 2019- 2025 without knowing which team one. Thus, the model uses 3% of the training data to test (~1 finals per test run x 400 test runs).