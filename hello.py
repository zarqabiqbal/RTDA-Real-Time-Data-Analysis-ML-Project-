#!/usr/bin/env python3
from flask import Flask, render_template, app, url_for,request
import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing
from textblob import TextBlob
import re
import pandas as pa
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

import time
import itertools

app=Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
	return render_template('index2.html')


@app.route('/index2')
def index2():
	return render_template('index.html')

@app.route('/layout')
def layout():
	return render_template('layout.html')


@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/Sentiment_Search', methods=['POST'])
def Sentiment_Search():
	search=request.form['search_Text']
	sid = SentimentIntensityAnalyzer()
	ss = sid.polarity_scores(search)
	neg = float(ss['neg']*100)
	neu = float(ss['neu']*100)
	pos = float(ss['pos']*100)
	compound =float(ss['compound']*100)
	ok=1
	return render_template("home.html",okk=ok,negg=neg,neuu=neu,poss=pos,comm=compound,srch=search)

@app.route('/facebook', methods=['POST'])
def facebook():
    try:
        driver = webdriver.Firefox()
        driver.get("https://www.facebook.com")
        wait = WebDriverWait(driver, 600)
        u_id = wait.until(EC.presence_of_element_located((By.XPATH,'//div[@class="_1k67 _cy7"]')))
        u_id.click()
        x=0
        while x<1000:
            driver.execute_script("window.scrollBy(0,2000)")
            time.sleep(1)
            x=x+50
        status=driver.find_elements_by_xpath('//div[@class="_1dwg _1w_m _q7o"]')
        stdetails=[]
        for i in status:
            stdetails.append(i.text)
        status_details=[]
        for i in stdetails:
                status_details.append(i.split())
        tokenized=list(itertools.chain.from_iterable(status_details))
        #remove punctuation from list
        tokenized=[i for i in tokenized if i.lower() not in stopwords.words('english')]
        sid = SentimentIntensityAnalyzer()
        neg=0
        neu=0
        pos=0
        compound=0
        for sentence in tokenized:
            ss = sid.polarity_scores(sentence)
            neg = neg+ float(ss['neg'])
            neu =  neu +float(ss['neu'])
            pos = pos + float(ss['pos'])
            compound = compound+float(ss['compound'])
        total=neg+neu+pos+compound
        negative=(neg/total)*100
        neutral=(neu/total)*100
        positive=(pos/total)*100
        compound=((compound/total)*100)
        if negative > neutral and negative > positive  and negative > compound:
            greatest=negative
            great="Highest Polarity is of Negative"
        if neutral > positive and neutral > negative and neutral > compound:
            greatest=neutral
            great="Highest Polarity is of Neutral"
        if positive > neutral and positive > negative and positive > compound:
            greatest=positive
            great="Highest Polarity is of Positive"
        if compound > neutral and compound > negative and compound > positive:
            greatest=positive
            great="Highest Polarity is of Compound"

        greatest= float("{0:.2f}".format(greatest))
        driver.close()
        return render_template('facebook_output.html',negg=negative,poss=positive,neuu=neutral,compp=compound,great_per=greatest,str_var=great)

    except:
        err=1
        titleshow="Some Error !! try again  ......."
        return render_template("whatsapp.html",error=titleshow,condition=err)



@app.route('/whatsappAnalysis', methods=['POST'])
def whatsappAnalysis():
    target=request.form['conversation_id']
    try:
        driver = webdriver.Firefox()
        driver.get("https://web.whatsapp.com/")
        wait = WebDriverWait(driver, 600)
        x_arg = '//span[contains(@title, '+ '"' +target + '"'+ ')]'
        person_title = wait.until(EC.presence_of_element_located((By.XPATH, x_arg)))
        person_title.click()
        x=-50
        chat=[]
        while x > -2000:
            element=driver.find_element_by_xpath("//div[@class='_9tCEa']")
            driver.execute_script("arguments[0].scrollIntoView(500);",element);
            x=x-100
            time.sleep(1)
        textget=driver.find_elements_by_class_name("selectable-text.invisible-space.copyable-text")
        print("Number of tweets extracted: {}.\n".format(len(textget)))
        for Text in textget:
            chat.append(Text.text)

        menu=driver.find_elements_by_class_name("rAUz7")
        menu[2].click()
        list=driver.find_elements_by_class_name("_10anr.vidHz._28zBA")
        list[5].click()
        a=len(chat)
        b=int(a/2)
        data=chat[b:a]
        sid = SentimentIntensityAnalyzer()
        neg=0
        neu=0
        pos=0
        compound=0
        for sentence in data:
            ss = sid.polarity_scores(sentence)
            neg = neg+ float(ss['neg'])
            neu =  neu +float(ss['neu'])
            pos = pos + float(ss['pos'])
            compound = compound+float(ss['compound'])
        total=neg+neu+pos+compound
        negative=(neg/total)*100
        neutral=(neu/total)*100
        positive=(pos/total)*100
        compound=((compound/total)*100)
        if negative > neutral and negative > positive  and negative > compound:
            greatest=negative
            great="Highest Polarity is of Negative"
        if neutral > positive and neutral > negative and neutral > compound:
            greatest=neutral
            great="Highest Polarity is of Neutral"
        if positive > neutral and positive > negative and positive > compound:
            greatest=positive
            great="Highest Polarity is of Positive"
        if compound > neutral and compound > negative and compound > positive:
            greatest=positive
            great="Highest Polarity is of Compound"

        greatest= float("{0:.2f}".format(greatest))
        driver.close()
        return render_template('facebook_output.html',negg=negative,poss=positive,neuu=neutral,compp=compound,great_per=greatest,str_var=great)

        print("ok")
    except:
        err=1
        titleshow="Some Error !! try again  ......."
        return render_template("facebook_output.html",error=titleshow,condition=err)





@app.route('/datacoming_twitter', methods=['POST'])
def data_twitter():
    try:
        CONSUMER_KEY    = '--'
        CONSUMER_SECRET = '--'
        ACCESS_TOKEN  = '--'
        ACCESS_SECRET = '--'
        def twitter_setup():
            auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
            auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
            api = tweepy.API(auth)
            return api
        # We create an extractor object:
        extractor = twitter_setup()
        SearchName=request.form['tw_username']
        tweets = extractor.user_timeline(screen_name="@"+SearchName, count=200)
        length_tweets=str(len(tweets))
        data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
        data['len']  = np.array([len(tweet.text) for tweet in tweets])
        data['ID']   = np.array([tweet.id for tweet in tweets])
        data['Date'] = np.array([tweet.created_at for tweet in tweets])
        data['Source'] = np.array([tweet.source for tweet in tweets])
        data['Likes']  = np.array([tweet.favorite_count for tweet in tweets])
        data['RTs']    = np.array([tweet.retweet_count for tweet in tweets])

        mean = np.mean(data['len'])
        fav_max = np.max(data['Likes'])
        rt_max  = np.max(data['RTs'])
        fav = data[data.Likes == fav_max].index[0]
        rt  = data[data.RTs == rt_max].index[0]
        liked_tweet=data['Tweets'][fav]
        retweets=data['Tweets'][rt]
        sources = []
        for source in data['Source']:
            if source not in sources:
                sources.append(source)

        def clean_tweet(tweet):
            """
            Utility function to clean the text in a tweet by removing
            links and special characters using regex.
            """
            return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
        def analize_sentiment(tweet):
            """
            Utility function to classify the polarity of a tweet
            using textblob
            """
            analysis = TextBlob(clean_tweet(tweet))
            if analysis.sentiment.polarity > 0:
                return 1
            elif analysis.sentiment.polarity == 0:
                return 0
            else:
                return -1
        data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])
        pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
        neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
        neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]
        pos_Percent=len(pos_tweets)/len(data['Tweets'])*100
        neu_Percent=len(neu_tweets)/len(data['Tweets'])*100
        neg_Percent=len(neg_tweets)/len(data['Tweets'])*100
        if pos_Percent > neu_Percent and pos_Percent > neg_Percent:
            greatest=pos_Percent
            great="Highest Polarity is of Positive"
        if neu_Percent > pos_Percent and neu_Percent > neg_Percent:
            greatest=neu_Percent
            great="Highest Polarity is of Neutral"
        if neg_Percent > pos_Percent and pos_Percent > neu_Percent:
            greatest=pos_Percent
            great="Highest Polarity is of Neagtive"
        greatest= float("{0:.2f}".format(greatest))

        return render_template('twitter_output.html',twit_src=sources,likeTweet=liked_tweet,retweet=retweets,pos=pos_Percent,neg=neg_Percent,neu=neu_Percent,great_per=greatest,str_var=great)
        print("ok")
    except:
        err=1
        titleshow="Some Error !! try again  ......."
        return render_template("twitter_output.html",error=titleshow,condition=err)

@app.route('/cancer')
def cancer():
    return render_template('cancer.html')

@app.route('/cancerPredict', methods=['POST'])
def cancerPredict():
    age=float(request.form['age'])
    gender=float(request.form['gender'])
    air=float(request.form['values'])
    alch=float(request.form['values1'])
    dust=float(request.form['values2'])
    occp=float(request.form['values3'])
    gene=float(request.form['values4'])
    ldesc=float(request.form['values5'])
    diet=float(request.form['values6'])
    obsty=float(request.form['values7'])
    smoke=float(request.form['values8'])
    psmoke=float(request.form['values9'])
    chest=float(request.form['values10'])
    cough=float(request.form['values11'])
    fatig=float(request.form['values12'])
    weight=float(request.form['values13'])
    breath=float(request.form['values14'])
    wheez=float(request.form['values15'])
    swallow=float(request.form['values16'])
    nails=float(request.form['values17'])
    cold=float(request.form['values18'])
    dcough=float(request.form['values19'])
    snore=float(request.form['values20'])
    data=pa.read_excel("cancer_patient_data_sets .xlsx").values
    #print(data)
    #print(data[0,1:24])
    train_data=data[0:998,1:24]
    train_target=data[0:998,24]
    '''print(train_target)
    test_data=data[999:,1:24]
    test_target=data[999:,24]
    print(test_target)'''
    clf=DecisionTreeClassifier()
    trained=clf.fit(train_data,train_target)
    clf1=SVC()
    trained1=clf1.fit(train_data,train_target)
    clf2=KNeighborsClassifier(n_neighbors=3)
    trained2=clf2.fit(train_data,train_target)

    test=[age,gender,air,alch,dust,occp,gene,ldesc,diet,obsty,smoke,psmoke,chest,cough,fatig,weight,breath,wheez,swallow,nails,cold,dcough,snore]
    #test=[34,1,2,3,4,5,6,7,6,5,4,3,2,1,2,3,4,5,2,3,5,2,3]
    predicted=trained.predict([test])
    predicted1=trained1.predict([test])
    predicted2=trained2.predict([test])

    print(predicted)
    print(predicted1)
    print(predicted2)

    #print(test_target)
    '''
    acc=accuracy_score(predicted,test_target)
    print(acc)
    acc1=accuracy_score(predicted1,test_target)
    print(acc)
    acc2=accuracy_score(predicted2,test_target)
    print(acc)
    '''
    #print(train_target)

    #print(age,gender,air,alch,dust,occp,gene,ldesc,diet,obsty,smoke,psmoke,chest,cough,fatig,weight,breath,wheez,swallow,nails,cold,dcough,snore)
    #return render_template("cancer.html",predicted=predicted,predicted1=predicted1,predicted2=predicted2)


if __name__ == '__main__':
    app.run("127.0.0.1",5000,debug=True)

