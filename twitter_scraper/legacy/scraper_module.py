import tweepy
import pandas as pd
import requests
import os

consumer_key = "ze86jWS8Mlac9oSH4LdVRHCvV"
consumer_secret = "QGnX0ATddMPDT9oaRh3RH9s2uXenD4KFpiosMEjC1F5rUMrCH0"
access_token = "366327791-wJ6fT2IKcG1xbK5OWoTLezJJbltVB2LbKLSAQIgw"
access_token_secret = "cza8zVMAZ62cf97cDLhJOGilMS07d1GAEVp067HbiM59r"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


def df_maker(hashtag, count=100):
    try:
        # Creation of query method using parameters
        tweets = tweepy.Cursor(api.search, q=hashtag).items(count)

        # Pulling information from tweets iterable object
        tweets_list = [[tweet.id, tweet.created_at,  tweet.text, tweet.entities['media'][0]['media_url']] for tweet in tweets
                       if ('media' in tweet.entities and 'RT' not in tweet.text[:2])]

        # Creation of dataframe from tweets list
        # Add or remove columns as you remove tweet information
        tweets_df = pd.DataFrame(tweets_list, columns =['id', 'date',  'text', 'url'])
    except BaseException as e:
        print("failed on_status,", str(e))
        time.sleep(3)
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