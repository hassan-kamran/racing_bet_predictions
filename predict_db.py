import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sys import argv

class Horse_model():
    def __init__(self,file):
        self.file = file
        self.df = pd.read_csv(f'./data/raw_data/{file}.csv')
        self.input = self.df.shape[1]

    def preprocess(self):
        self.df = self.df[['StartingOdds','RecentWinPercent','Class','laststart']]
        median = self.df.median()
        self.df = self.df.fillna(median)
        self.df_scaled = pd.DataFrame(StandardScaler().fit_transform(self.df), columns=self.df.columns)
    
    def create_nn_model(self):
        input = tf.keras.Input(shape=(4,))
        x = tf.keras.layers.Dense(8, activation='relu')(input)
        x = tf.keras.layers.Dense(4, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        self.model = tf.keras.Model(input, output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.load_weights('./checkpoints/my_checkpoint')

    def predict(self):
        self.pred = self.model.predict(self.df_scaled)
        self.pred = pd.DataFrame(self.pred, columns=['Winners'])
        self.pred.to_csv(f'./predictions/{self.file}_pred_only.csv', index=False)
        self.df['Win_pred'] = self.pred
        self.df.to_csv(f'./predictions/{self.file}_pred.csv',index=False)

if __name__ == '__main__':
    horse = Horse_model(argv[1])
    horse.preprocess()
    horse.create_nn_model()
    horse.predict()