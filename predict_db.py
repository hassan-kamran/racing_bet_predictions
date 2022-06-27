import tensorflow as tf
import pandas as pd
from sys import argv
from sklearn.preprocessing import StandardScaler

class Horse_model():
    def __init__(self,file):
        self.file = file
        self.df = pd.read_csv(f'./data/raw_data/{file}.csv')

    def preprocess(self):
        self.df_features = self.df[['StartingOdds','RecentWinPercent','Class','laststart']]
        self.input = self.df_features.shape[1]
        self.df_identifier = self.df[['DayCalender','RaceName','Venue','RaceDistance','HorseName']]
        median = self.df_features.median()
        self.df_features = self.df_features.fillna(median)
        self.df_features_scaled = pd.DataFrame(StandardScaler().fit_transform(self.df_features), columns=self.df_features.columns)

    def create_nn_model(self):
        self.model = tf.keras.models.load_model('./model/nn_model.h5')

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
    for args in argv[1:]:
        horse = Horse_model(args)
        horse.preprocess()
        horse.create_nn_model()
        horse.predict()
        horse.format()