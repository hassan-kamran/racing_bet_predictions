import pandas as pd
from sys import argv

def write_grp_to_file(info, grp, file_name):
    with open(f'./predictions/{file_name}_formated.txt', 'a') as f:
        f.write(info+'\n')
        count = 1
        for row in grp.itertuples(index=False):
            row = dict(row._asdict())
            f.write(f'{count}.Horse Name:{row["HorseName"]},Win Probability %:{row["Winners_Probability"]}\n')
            count += 1
        f.write('\n')

def format(file):
    data = pd.read_csv(f'./predictions/{file}.csv')
    grouped_data = data.groupby(['DayCalender','RaceName','Venue','RaceDistance'], as_index=False)
    for group in grouped_data:
        info = f'Date:{group[0][0]}\nRace Name:{group[0][1]}\nVenue:{group[0][2]}\nRace Distance:{group[0][3]}\n'
        write_grp_to_file(info, group[1], file)

if __name__ == '__main__':
    format(argv[1])