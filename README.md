🚀 **[Live Demo](https://premier-league-analytics-vadmjhfwbgn3ceuzj5vjkj.streamlit.app/)**

[![Open Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://premier-league-analytics-vadmjhfwbgn3ceuzj5vjkj.streamlit.app/)
\# ⚽ Premier League Analytics Lab



An interactive football analytics platform built with \*\*Python, Streamlit, Pandas, Scikit-learn and DuckDB\*\*.



The project combines Premier League player statistics, market values, club information and machine-learning based valuation analysis into a modern football analytics dashboard.



\---



\## 🚀 Features



\### 🏠 Analytics Dashboard

\- Premier League overview

\- Market-value leaders

\- Player statistics

\- Club explorer

\- Interactive navigation



\### 🏟️ Club Analytics

\- Browse Premier League clubs

\- Club logos

\- Club player lists

\- Squad statistics

\- Total squad market value



\### 👤 Player Analytics

\- Player profiles

\- Player photographs

\- Position and playing role

\- Age

\- Appearances

\- Minutes played

\- Goals

\- Assists

\- Goals per 90

\- Assists per 90

\- Market value



\### 🤖 Machine Learning

The project includes a player market-value prediction pipeline using:



\- Linear Regression

\- Random Forest

\- Extra Trees

\- Gradient Boosting

\- Feature preprocessing

\- One-hot encoding

\- Missing-value imputation

\- Log-transformed market-value target



The model produces a \*\*model-implied player value estimate\*\* that can be compared with the listed market value.



> The prediction should be interpreted as a statistical estimate from the available features, not as an objective valuation of a player.



\---



\## 📊 Machine Learning Results



The initial model comparison was performed on the project's Premier League player dataset.



| Model | MAE | RMSE | R² |

|---|---:|---:|---:|

| Linear Regression | €8.62M | €14.84M | 0.666 |

| Random Forest | €10.80M | €19.39M | 0.430 |

| Extra Trees | €9.52M | €17.95M | 0.512 |

| Gradient Boosting | €8.23M | €15.67M | 0.628 |



For this experiment, \*\*Gradient Boosting had the lowest MAE\*\* among the tested models.



These results are based on the project's original train/test split and should not be interpreted as a guarantee of real-world transfer-market prediction accuracy.



\---



\## 🧠 Features Used for Valuation



The valuation model uses performance and player information such as:



\- Age

\- Appearances

\- Minutes played

\- Goals

\- Assists

\- Goals per 90

\- Assists per 90

\- Minutes per appearance

\- Position

\- Sub-position

\- Club



The target variable is the player's market value.



\---



\## 🗂️ Project Structure



```text

Premier-League-Analytics/

│

├── app.py

├── build\_dataset.py

├── train.py

├── train\_model.py

├── train\_models.py

├── analyze\_players.py

├── explore\_data.py

├── club\_assets.py

│

├── data/

│   ├── premier\_league\_players.csv

│   ├── best\_model\_predictions.csv

│   ├── model\_predictions.csv

│   └── player\_value\_analysis.csv

│

├── models/

│   ├── best\_player\_value\_model.pkl

│   └── player\_value\_model.pkl

│

├── assets/

│   └── clubs/

│

└── .gitignore

