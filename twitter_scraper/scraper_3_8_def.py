import pandas as pd
import requests
import os, sys
import time
import snscrape.modules.twitter as sntwitter

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
        lista=[]
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(hashtag).get_items()) :
                if i > count :
                    break
                if tweet.media!=None:
                    try: 
                        media=((tweet.media[0].previewUrl).replace('?format=', '.').replace('&name=small', ''))
                    except:
                        media=(tweet.media[0].thumbnailUrl)
                    lista.append([tweet.id, tweet.date, tweet.renderedContent, media])

        tweets_df=pd.DataFrame(lista, columns =['id', 'date',  'text', 'url'])
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
    lista_hashtag = ['donmatteo12']#'LiveNoneLadUrso','DomenicaLive','ballandoconlestelle',
                    #'TEMPTATIONISLAND', 'domenicain','uominiedonne','tusiquevales','XF2020',
                    #'leiene','Pomeriggio5', 'lavitaindiretta', 'immatataranni','lallieva3','CTCF',
                    #'taleequaleshow','ilcollegio', 'IoTiCerchero','QuartaRepubblica','report', 
                    #'GFVIP', 'tikitaka','lallieva2', , 'sanremo2020', 'masterchefit',
                    #'amici19', 'sanremo19','docnelletuemani','chediociaiuti','nondirloalmiocapo2', 'lamicageniale',
                    #'upas'
    count = 5000

    for hashtag in lista_hashtag:
        print(f'scraping for {hashtag}')
        scraper = TwitterScraper(hashtag=hashtag, count=count, output_folder=output_folder, image_folder=image_folder)
        scraper.scrape_tweets()
        time.sleep(10)
    print('----FINISHED!----')
        # print(os.path.dirname(os.path.abspath(sys.argv[0])))