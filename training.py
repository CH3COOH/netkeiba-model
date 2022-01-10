import sqlite3
#from optuna.integration import lightgbm as lgb
import lightgbm as lgb
from numpy import NAN, NaN
from sklearn.model_selection import train_test_split
import pandas as pd

# race.dbのパスを指定する
dbPath = '..//netkeiba-scraper/race.db'

# データの取得と利用するデータの整形
conn = sqlite3.connect(dbPath)
df = pd.read_sql_query('SELECT * FROM feature', conn)
conn.close()

df = df.reindex(
    columns=[
        'horse_id', 'jockey_id', 'trainer_id', 'horse_number', 'odds', 'order_of_finish'
    ]
)

# オッズ・順位が入力されていない行は除外する
df = df.dropna(subset=['odds'])
df = df.dropna(subset=['order_of_finish'])

# lightgbmは int か float か bool以外の値は扱えないので補正する
df["horse_id"] = df["horse_id"].astype(int)
df["jockey_id"] = df["jockey_id"].astype(int)
df["trainer_id"] = df["trainer_id"].astype(int)

# print(df)
# print(df.head())
print(df.isnull().sum())
print(df.dtypes)

# モデルを作成する

# バックテストで検証する

# 解説用にわかりやすいグラフを表示する