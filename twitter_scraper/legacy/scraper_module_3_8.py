import tweepy
import pandas as pd
import requests
import os
import time

consumer_key = "ze86jWS8Mlac9oSH4LdVRHCvV"
consumer_secret = "QGnX0ATddMPDT9oaRh3RH9s2uXenD4KFpiosMEjC1F5rUMrCH0"
access_token = "366327791-wJ6fT2IKcG1xbK5OWoTLezJJbltVB2LbKLSAQIgw"
access_token_secret = "cza8zVMAZ62cf97cDLhJOGilMS07d1GAEVp067HbiM59r"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


def df_maker(hashtag, count=100):
    lista=[]
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(keyword).get_items()) :
            if i > maxTweets :
                break
            if tweet.media!=None:
                try: 
                    media=((tweet.media[0].previewUrl).replace('?format=', '.').replace('&name=small', ''))
                except:
                    media=(tweet.media[0].thumbnailUrl)
                lista.append([tweet.id, tweet.date, tweet.renderedContent, media])
    tweets_df=pd.DataFrame(lista, columns =['id', 'date',  'text', 'url'])
    return(tweets_df)

def save_url(index, url, path):
    try:
        os.makedirs(path)
    except:
        pass
    response = requests.get(url)
    file = open(f"{path}/{index}.png", "wb")
    file.write(response.content)
    file.close()

def image_saver(df, path):
    for idx, row in df.iterrows():
        index=row['id']
        url=row['url']
        save_url(index, url, path)

def scraper(hashtag, count):
    try:
        os.makedirs(f"data/{hashtag}")
    except:
        pass
    dataframe=df_maker(hashtag, count)
    image_saver(dataframe, f"data/{hashtag}/images")
    try:
        old=pd.read_csv(f"data/{hashtag}/data.csv")
        dataframe=old.append(dataframe)
        dataframe=dataframe.drop_duplicates(subset='id')
    except:
        pass
    dataframe.to_csv(path_or_buf=f"data/{hashtag}/data.csv", index=False)
    return 'tutto ok'


scraper('GerryScotti', 1000)