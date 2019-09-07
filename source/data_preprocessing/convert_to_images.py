import pandas as pd
import numpy as np
import cv2
IMAGE_TYPE=1
DATASET_PATH='../../data/raw/fer2013.csv'
PROCESSED_DATA_EMOTION_PATH='../../data/processed/emotion.pkl'
PROCESSED_DATA_IMAGE_PATH='../../data/processed/images.pkl'

IMAGE_HEIGHT=48
IMAGE_WIDTH=48
CHUNK_SIZE=10000
CHUNK_list=[]

file_reader=pd.read_csv(DATASET_PATH,chunksize=CHUNK_SIZE)

for chunk in file_reader :
    CHUNK_list.append(chunk)
df=pd.concat(CHUNK_list,axis=0)


def str_to_img(row):
    return np.array([int (i) for i in row.split(' ')],dtype=np.uint8).reshape(IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_TYPE)
df['pixels']=df['pixels'].apply(lambda x:str_to_img(x))

pd.DataFrame(df['emotion']).to_pickle(PROCESSED_DATA_EMOTION_PATH)
pd.DataFrame(df['pixels']).to_pickle(PROCESSED_DATA_IMAGE_PATH)
#cv2.imshow('test',sample)
#cv2.waitKey(0)
#print(df['emotion'][8])