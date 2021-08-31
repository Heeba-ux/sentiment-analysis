# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:59:18 2021

@author: Heeba
"""


import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer

web_url = 'https://finviz.com/quote.ashx?t='

news_tables = {}
tickers = ['AMZN', 'GOOG', 'TSLA']

for tick in tickers:
    url = web_url + tick
    req = Request(url=url,headers={"User-Agent": "Chrome"}) 
    response = urlopen(req)    
    html = BeautifulSoup(response,"html.parser")
    news_table = html.find(id='news-table')
    news_tables[tick] = news_table

amazon = news_tables['AMZN']
amazon_tr = amazon.findAll('tr')

for x, table_row in enumerate(amazon_tr):
    a_text = table_row.a.text
    td_text = table_row.td.text
    print(a_text)
    print(td_text)
    if x == 3:
        break

news_list = []

for file_name, news_table in news_tables.items():
    for i in news_table.findAll('tr'):
        
        text = i.a.get_text() 
        
        date_scrape = i.td.text.split()

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        else:
            date = date_scrape[0]
            time = date_scrape[1]

        tick = file_name.split('_')[0]
        
        news_list.append([tick, date, time, text])
    
news_list


vader = SentimentIntensityAnalyzer()
columns = ['ticker', 'date', 'time', 'headline']
news_df = pd.DataFrame(news_list, columns=columns)
scores = news_df['headline'].apply(vader.polarity_scores).tolist()
scores_df = pd.DataFrame(scores)
news_df = news_df.join(scores_df, rsuffix='_right')
news_df['date'] = pd.to_datetime(news_df.date).dt.date

print(news_df)


plt.rcParams['figure.figsize'] = [10, 6]
mean_scores = news_df.groupby(['ticker','date']).mean()
mean_scores = mean_scores.unstack()
mean_scores = mean_scores.xs('compound', axis="columns").transpose()
mean_scores.plot(kind = 'bar')
plt.grid()
plt.show()

#mean
Am = news_df[news_df['ticker'] == 'AMZN']
gg = news_df[news_df['ticker'] == 'GOOG']
ts = news_df[news_df['ticker'] == 'TSLA']

#Am.describe()
#gg.describe()
#ts.describe()
a = Am.mean()
b = gg.mean()
c = ts.mean()
means = Am.mean(),gg.mean(),ts.mean()
means

#for amazon
#fig = plt.figure(figsize =(10, 7))
#labels = ['negative','neutral','positve','compound']
#plt.pie(a, labels = labels)
#plt.title("amazon", bbox={'facecolor':'0.8', 'pad':5})
#plt.show()
#plt.figure(0)
#for google
#fig = plt.figure(figsize =(10, 7))
#labels = ['negative','neutral','positve','compound']
#plt.pie(b, labels = labels)
#plt.title("google", bbox={'facecolor':'0.8', 'pad':5})
#plt.show()
#plt.figure(1)
#for tesla
#fig = plt.figure(figsize =(10, 7))
#labels = ['negative','neutral','positve','compound']
#plt.pie(c, labels = labels)
#plt.title("tesla", bbox={'facecolor':'0.8', 'pad':5})
#plt.figure(2)



labels = ['negative','neutral','positve','compound']
sizes = [a]
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0, 0.1, 0, 0)
#add colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.title("amazon")
plt.tight_layout()
plt.show()


labels = ['negative','neutral','positve','compound']
sizes = [b]
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0, 0.1, 0, 0)
#add colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.title("google")
plt.tight_layout()
plt.show()

labels = ['negative','neutral','positve','compound']
sizes = [c]
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0, 0.1, 0, 0)
#add colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.title("tesla")
plt.tight_layout()
plt.show()

""""
import seaborn as sns
sns.distplot(a, hist=False, rug=True)
sns.distplot(b, hist=False, rug=True)
sns.distplot(c, hist=False, rug=True)
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,6))
fig.legend(labels=['AMZN','GOOG','TESL'])
plt.show()
sns.distplot()
"""

#news_df.to_csv("analysis.csv")
#os.chdir("C:/Users/DELL/Downloads")
#os.getcwd()
import pickle
pickle.dump(open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
