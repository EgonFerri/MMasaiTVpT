import tweepy
import pandas as pd
import requests
import os, sys
import time

consumer_key = "ze86jWS8Mlac9oSH4LdVRHCvV"
consumer_secret = "QGnX0ATddMPDT9oaRh3RH9s2uXenD4KFpiosMEjC1F5rUMrCH0"
access_token = "366327791-wJ6fT2IKcG1xbK5OWoTLezJJbltVB2LbKLSAQIgw"
access_token_secret = "cza8zVMAZ62cf97cDLhJOGilMS07d1GAEVp067HbiM59r"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


class TwitterScraper:

    def __init__(self, hashtag, count, output_folder, image_folder):
        self.hashtag = hashtag
        self.count = count
        self.output_folder = output_folder + hashtag
        self.image_folder = self.output_folder + image_folder
        print(self.image_folder)
        try:
            os.makedirs(self.output_folder)
            os.makedirs(self.image_folder)
            print('*** Folders created! ***')
        except:
            pass

    def df_maker(self):
        try:
            # Creation of query method using parameters
            tweets = tweepy.Cursor(api.search, q=self.hashtag, result_type='recent').items(self.count)

            # Pulling information from tweets iterable object
            tweets_list = [[tweet.id, tweet.created_at, tweet.text, tweet.entities['media'][0]['media_url']] for tweet
                           in tweets
                           if ('media' in tweet.entities and 'RT' not in tweet.text[:2])]

            # Creation of dataframe from tweets list
            # Add or remove columns as you remove tweet information
            tweets_df = pd.DataFrame(tweets_list, columns=['id', 'date', 'text', 'url'])
        except BaseException as e:
            print("failed on_status,", str(e))
            time.sleep(3)
        return tweets_df

    def image_saver(self, df):
        for idx, row in df.iterrows():
            index = row['id']
            url = row['url']
            response = requests.get(url)
            with open(self.image_folder + f'{index}.png', "wb") as f:
                f.write(response.content)

    def scrape_tweets(self):
        dataframe = self.df_maker()
        if len(dataframe)>0:
            self.image_saver(dataframe)
            try:
                old = pd.read_csv(self.output_folder + '/data.csv')
                dataframe = old.append(dataframe)
                dataframe = dataframe.drop_duplicates(subset='id')
            except:
                pass
            print(len(dataframe))
            dataframe.to_csv(path_or_buf=self.output_folder + '/data.csv', index=False)
            print('** Data written! ** ')
        else:
            print('I found no data')


if __name__ == '__main__':

    output_folder = 'C:/Users/Egon/projects/visual_sentiment_analysis/twitter_scraper/data/' #os.getcwd()  + '/data/'
    # output_folder = os.getcwd() + '/data/'
    image_folder = '/images/'
    lista_hashtag = ['IoTiCerchero','QuartaRepubblica','report', 'GFVIP', 'tikitaka']# 
    count = 100

    for hashtag in lista_hashtag:
        print(f'scraping for {hashtag}')
        scraper = TwitterScraper(hashtag=hashtag, count=count, output_folder=output_folder, image_folder=image_folder)
        scraper.scrape_tweets()
        # print(os.path.dirname(os.path.abspath(sys.argv[0])))