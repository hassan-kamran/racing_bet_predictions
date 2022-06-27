import pandas as pd
from sys import argv
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class Horse_model():
    def __init__(self,file):
        self.file = file
        
    def load_training_data(self):
        loaded_files = []
        for file in self.file:
            df = pd.read_csv(f'./data/raw_data/{file}.csv')
            df = df[['Winner','StartingOdds','RecentWinPercent','Class','laststart']]
            loaded_files.append(df)
        self.df = pd.concat(loaded_files, axis=0)
        self.df.fillna(self.df.median(), inplace=True)
        self.df.to_csv('./data/horse_data.csv', index=False)
        self.x = self.df.drop(['Winner'], axis=1)
        self.x = pd.DataFrame(StandardScaler().fit_transform(self.x), columns=self.x.columns)
        self.y = self.df['Winner']

    def create_nn_model(self):
        input = tf.keras.Input(shape=(self.x.shape[1],))
        m = tf.keras.layers.Dense(8, activation='relu')(input)
        m = tf.keras.layers.Dense(4, activation='relu')(m)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(m)
        self.nn_model = tf.keras.Model(input, output)
        self.nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_nn_model(self):
        self.nn_model.fit(self.x, self.y, epochs=10, verbose=1)
        self.nn_model.save('./model/nn_model.h5')

    def preprocess(self):
        self.df = pd.read_csv(f'./data/raw_data/{self.file}.csv')
        self.df_features = self.df[['StartingOdds','RecentWinPercent','Class','laststart']]
        self.input = self.df_features.shape[1]
        self.df_identifier = self.df[['DayCalender','RaceName','Venue','RaceDistance','HorseName']]
        self.df_features.fillna(self.df_features.median(), inplace=True)
        self.df_features_scaled = pd.DataFrame(StandardScaler().fit_transform(self.df_features), columns=self.df_features.columns)

    def load_nn_model(self):
        self.nn_model = tf.keras.models.load_model('./model/nn_model.h5')

    def predict(self):
        self.pred = self.model.predict(self.df_features_scaled)
        self.pred = pd.DataFrame(self.pred, columns=['Winners_Probability'])
        self.pred = self.pred*100
        self.df_combined = pd.concat([self.df_identifier, self.df_features, self.pred], axis=1)
        self.df_combined.to_csv(f'./predictions/{self.file}_pred.csv',index=False)

    def format(self):
        grouped_data = self.df_combined.groupby(['DayCalender','RaceName','Venue','RaceDistance'], as_index=False)
        for group in grouped_data:
            info = f'Date:{group[0][0]}\nRace Name:{group[0][1]}\nVenue:{group[0][2]}\nRace Distance:{group[0][3]}\n'
            with open(f'./predictions/{self.file}_pred_formated.txt', 'a') as f:
                f.write(info+'\n')
                count = 1
                for row in group[1].itertuples(index=False):
                    row = dict(row._asdict())
                    f.write(f'{count}.Horse Name:{row["HorseName"]},Win Probability %:{row["Winners_Probability"]}\n')
                    count += 1
                f.write('\n')
                
if __name__ == '__main__':
    if argv[1] == '-t':
        file = []
        for args in argv[2:]:
            file.append(args)
        horse = Horse_model(file)
        horse.load_training_data()
        horse.create_nn_model()
        horse.train_nn_model()
    elif argv[1] == '-p':
        for args in argv[2:]:
            horse = Horse_model(args)
            horse.preprocess()
            horse.load_nn_model()
            horse.predict()
            horse.format()
    else:
        print('Invalid flag')