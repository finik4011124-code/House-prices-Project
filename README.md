# House Prices 


## Usage

python -m venv .venv
venv\Scripts\activate  
pip install -r requirements.txt
python main.py


## Structure

house-prices/
├── data/                     # train.csv, test.csv 
├── notebooks/
│   ├── EDA.ipynb             
│   └── experiments.ipynb    # preprocessing, modeling
├── results/
│   └── comparison_table.csv
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── model.py
│   └── utils.py
├── config.yaml               # model hyperparameters
├── main.py                   
└── requirements.txt

## Results

| Experiment               | RMSE   | Notes                                      |
|--------------------------|--------|--------------------------------------------|
| LinearRegression         | 0.1309 | after preprocessing, no FE                 |
| LinearRegression + FE    | 0.1318 | after preprocessing + FE                   |
| RandomForest (GridSearch)| 0.1348 | n_estimators=300, max_depth=None           |
| XGBoost (Optuna)         | 0.1136 | n_estimators=507, lr=0.049, max_depth=3    |
| LightGBM (Optuna)        | 0.1188 | n_estimators=1080, lr=0.049, max_depth=3   |
| CatBoost (Optuna)        | 0.1121 | iterations=992, lr=0.030, depth=5          |
| Voting (XGB+LGB+CAT)     | 0.1120 | VotingRegressor with best params           |
| NeuralNet (skorch)       | 0.2123 | lr=0.0099, batch=64, epochs=187            |

