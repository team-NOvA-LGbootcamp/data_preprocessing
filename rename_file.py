import os
from os import listdir
from os.path import isfile, join
import csv
import pandas as pd
import datetime


def age2str(x):
    if x >0: return str(int(x))
    else: return 'unknown'
    
def gender2str(x):
    if x=='Boy': return '0'
    elif x=='Girl': return '1'
    else: return 'unknown'



result_dir_path = "./results"
IMAGE_FILES = [f for f in listdir(result_dir_path) if isfile(join(result_dir_path, f))]

df = pd.read_csv("./Data_label.csv")
df['Race'] = '2'
df['Age'] = df['Age'].apply(age2str)
df['Gender'] = df['Gender'].apply(gender2str)


for idx, img in enumerate(IMAGE_FILES):
    img_name = img[:-4].strip()
    dff = df[df["Img"]==img_name]

    if len(dff)>0:
        now = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        timestamp = f"{now}{idx}"
        new_fname = f"{dff.iloc[0,1]}_{dff.iloc[0,2]}_2_{timestamp}"   

        # print(new_fname)
        os.rename("./results/"+img, "./final_results/"+new_fname+'.jpg')
    