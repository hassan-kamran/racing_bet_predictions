import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sys import argv

def one_hot_encoding(df, columns):
    df = df.copy()
    for col, pref in columns.items():
        dummy = pd.get_dummies(df[col], prefix=pref)
        dummy.astype(bool)
        df = pd.concat([df, dummy], axis=1)
        df = df.drop(col, axis=1)
    return df

def preprocessing(df):
    df = df.copy()
    try:
        df = one_hot_encoding(df, {'Row':'Row','Sex': 'Sex'})
    except:
        df = one_hot_encoding(df, {'Row':'Row','sex': 'Sex'})
    df = df[['StartingOdds','RecentPlacePercent','RecentWinPercent','Class','Row_1','Sex_COLT','Sex_HORSE','laststart']]
    df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    return df

def create_model(df):
    input = tf.keras.Input(shape=(df.shape[1],))
    x = tf.keras.layers.Dense(128, activation='relu')(input)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    nn_model = tf.keras.Model(input, output)
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.load_weights('./checkpoints/my_checkpoint')
    return nn_model

def predict_db(df,nn_model):
    winners = nn_model.predict(df)
    return winners

if __name__ == '__main__':
    df = pd.read_csv(f'./data/{argv[1]}.csv')
    df = preprocessing(df)
    nn_model = create_model(df)
    winners = predict_db(df,nn_model)
    winners = pd.DataFrame(winners, columns=['Winners'])
    winners.to_csv(f'./predictions/winners_{argv[1]}.csv', index=False)