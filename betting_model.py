import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def horses_preprocess(dfs):
    dfs_processed = []
    for df in dfs:
        df = df.copy()
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

corr = horses.corr().sort_values(by='Winner', ascending=False)
winner_corr = corr[['Winner']]
fig, ax = plt.subplots(figsize=(5,5), dpi=150)
annot_kws = {"ha": 'center',"va": 'top', "size":5, "color": "black"}
ax = sns.heatmap(winner_corr, annot=True, vmax=winner_corr.max(), vmin=winner_corr.min(), square=False, cbar=True, cmap='RdBu_r', linewidths=0.5, linecolor='black', annot_kws=annot_kws)
ax.set_title('Winner Correlation')
plt.show()

horses.info(verbose=True, show_counts=True)

x = horses.drop(['Winner'], axis=1)
y = horses['Winner']
x = pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=53)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

lr = LogisticRegression(solver='newton-cg').fit(x_train, y_train)
print(f'Logistic Regression: {round(lr.score(x_test, y_test)*100)}%')

input = tf.keras.Input(shape=(x.shape[1],))
x = tf.keras.layers.Dense(8, activation='relu')(input)
x = tf.keras.layers.Dense(4, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
nn_model = tf.keras.Model(input, output)
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=1)

nn_model.evaluate(x_test, y_test)

nn_model.save_weights('./checkpoints/my_checkpoint')