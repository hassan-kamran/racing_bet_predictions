import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def horses_preprocess(dfs):
    dfs_processed = []
    for df in dfs:
        df = df[['Winner','StartingOdds','RecentWinPercent','Class','laststart']]
        median = df.median()
        df.fillna(median, inplace=True)
        dfs_processed.append(df)
    return pd.concat(dfs_processed, axis=0)

horses0 = pd.read_csv('./data/raw_data/horses.csv')
horses1 = pd.read_csv('./data/raw_data/horses1.csv')
horses2 = pd.read_csv('./data/raw_data/horses2.csv')
horses = horses_preprocess([horses1, horses2, horses0])
horses.to_csv('./data/horse_data.csv', index=False)

x = horses.drop(['Winner'], axis=1)
x = pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns)
y = horses['Winner']

input = tf.keras.Input(shape=(x.shape[1],))
m = tf.keras.layers.Dense(8, activation='relu')(input)
m = tf.keras.layers.Dense(4, activation='relu')(m)
output = tf.keras.layers.Dense(1, activation='sigmoid')(m)
nn_model = tf.keras.Model(input, output)
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(x, y, epochs=10)
nn_model.save('./model/nn_model.h5')