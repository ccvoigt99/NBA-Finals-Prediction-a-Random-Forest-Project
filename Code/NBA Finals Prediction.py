### Project 4 - AAE 718
import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
from kagglehub import KaggleDatasetAdapter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

#Importing Data/Merging
def load_data():
    filepath = "championsdata.csv"
    champdf = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "daverosenman/nba-finals-team-stats",
        filepath,
    )

    filepath2 = "runnerupsdata.csv"
    rudf = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "daverosenman/nba-finals-team-stats",
        filepath2,
    )

    # Merge on Year 
    merged_df = pd.merge(champdf, rudf, on=['Year', 'Game'], suffixes=('_champ', '_ru'))

    # Create difference variables
    merged_df['FGPDiff'] = merged_df['FGP_champ'] - merged_df['FGP_ru']
    merged_df['TRBDiff'] = merged_df['TRB_champ'] - merged_df['TRB_ru']
    merged_df['PTSDiff'] = merged_df['PTS_champ'] - merged_df['PTS_ru']

    # Reshaping into long format
    champ = merged_df[[col for col in merged_df.columns if '_champ' in col or col in ['Year', 'Game']]+['FGPDiff','TRBDiff', 'PTSDiff']]
    champ = champ.rename(columns=lambda x: x.replace('_champ', ''))
    champ['Winner'] = 1

    ru = merged_df[[col for col in merged_df.columns if '_ru' in col or col in ['Year', 'Game']]+['FGPDiff', 'TRBDiff', 'PTSDiff']]
    ru = ru.rename(columns=lambda x: x.replace('_ru', ''))
    ru['Winner'] = 0

    long_df = pd.concat([champ, ru], ignore_index=True)
    long_df = long_df.replace(np.nan, 0)
    return long_df

df = (load_data())

# I create five dataframes for Game 1, Games 1 and 2, and Games 1, 2, and 3. This is relevant for the random forest models.
game1 = df[df['Game'] == 1]
game12 = df[df['Game'].isin([1, 2])]
game123 = df[df['Game'].isin([1, 2, 3])]


# Now, I implement 3 different random forests, using data from Game 1, Games 1 and 2, and Games 1, 2, and 3.

def rf(game):

    features = ['Home', 'Game', 'AST','FTA', 'PTSDiff', 'FGPDiff', 'TRBDiff' ]
    X = game[features]
    y = game['Winner']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=42)

    model = RandomForestClassifier(n_estimators=400, random_state=42, max_depth=8, max_features= 4)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Training accuracy
    trainingaccuracy = accuracy_score(y_train, model.predict(X_train))
    print(f"Training accuracy: {trainingaccuracy:.2f}") #this prints the training accuracy

    # Test accuracy
        # Testing accuracy
    testaccuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Testing accuracy: {testaccuracy:.2f}")

    return model

def load_testdata():

    filepath3 = "championsdata2019-2025g3.csv"

    testchampdf = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "charlesvoigt/nba-finals-2019-2025g3",
        filepath3,
    )

    filepath4 = "runnerupsdata2019-2025g3.csv"

    testrudf = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "charlesvoigt/nba-finals-2019-2025g3",
        filepath4,
    )
    # Merge on Year 
    merged_df = pd.merge(testchampdf, testrudf, on=['Year', 'Game'], suffixes=('_champ', '_ru'))

    # Create difference variables
    merged_df['FGPDiff'] = merged_df['FGP_champ'] - merged_df['FGP_ru']
    merged_df['TRBDiff'] = merged_df['TRB_champ'] - merged_df['TRB_ru']
    merged_df['PTSDiff'] = merged_df['PTS_champ'] - merged_df['PTS_ru']

    # Reshaping into long format
    champ = merged_df[[col for col in merged_df.columns if '_champ' in col or col in ['Year', 'Game']]+['FGPDiff','TRBDiff', 'PTSDiff']]
    champ = champ.rename(columns=lambda x: x.replace('_champ', ''))
 
    

    ru = merged_df[[col for col in merged_df.columns if '_ru' in col or col in ['Year', 'Game']]+['FGPDiff', 'TRBDiff', 'PTSDiff']]
    ru = ru.rename(columns=lambda x: x.replace('_ru', ''))
    

    long_df = pd.concat([champ, ru], ignore_index=True)
    long_df = long_df.replace(np.nan, 0)
    
    return long_df

def predict_rf_on_test(model, test_df):
    features = ['Home', 'Game', 'AST', 'FTA', 'PTSDiff', 'FGPDiff', 'TRBDiff']
    X_test = test_df[features]

    y_pred = model.predict(X_test)
    test_df['Predicted_Winner'] = y_pred

    # Only print predicted champion for each year (no accuracy calculation)
    for year in sorted(test_df['Year'].unique()):
        winner_row = test_df[(test_df['Year'] == year) & (test_df['Predicted_Winner'] == 1)]
        print(f"Predicted champion for {year}: {winner_row['Team'].values[0]}")
       

    return test_df

game = game123 # Change this to the desired game dataframe (game1, game12, or game123)

#rf(game) #This will train the random forest model(s), and allow us to see training and test accuracy.
#predict_rf_on_test(rf(game), load_testdata()) # This will predict the winners for the test data (NBA finals from 2019 - 2025 Game 3) using the random forest models.