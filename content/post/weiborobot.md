+++
date = "2016-07-02T16:17:24+02:00"
description = ""
draft = false
tags = ["SinaWeibo API", "Weather"]
title = "Turn Your Social Account into a Weather Robot"
topics = []
slug = "weibo_robot"

+++
#### Demo
This is my recent play-for-fun, a tiny trick developing a weatherman robot on SNS.  

E.g.
<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/weibopost.png" width = "700" height = "200"/>  
</div>  
This robot will update the HK local weather everyday at 20:00 EDT and post it on my SinaWeibo account.

#### Checklist  

- A SinaWeibo developer account to use its post API (register [here](http://open.weibo.com/wiki/首页). You will need either a personal webpage or an application under development for registration).
- A [WorldWeatherOnline](http://developer.worldweatheronline.com/api/) API to update weather. They provide free API to fetch the weather worldwide and pack it in popular formats like xml, json or csv.
- A Linux server.  

#### Pipeline  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/pipeline.png" width = "600" height = "250"/>  
</div>  

**WWO API**

Use urllib to run GET request to WWO, get back weather information in json  

```
import urllib2
url = 'http://api.worldweatheronline.com/premium/v1/weather.ashx?key=*******************&q=HongKong&format=json&num_of_days=1&date=today&mca=no&fx24=yes&showlocaltime=yes'
response = urllib2.urlopen(url).read()
```  

**json decode and extraction**

In python one can easily use json library for json decoding, then extract information interested.

```
import json
js = json.loads(response)
w = 'Good morning! Weather robot online. Current Query: ' + js['data']['request'][0]['query'] + '.\n'
w += 'Current weather condition: ' + js['data']['current_condition'][0]['weatherDesc'][0]['value'] + ', '
w += 'Feels like ' + js['data']['current_condition'][0]['FeelsLikeC'] + ' degree Celsius' + '.'
```

**Weibo post**

Use Weibo API to publish a post.

```
client.post('statuses/update', status=w)
```  

to post. Make sure to figure out the OAuth2.0 authorization before using this API.  

**Server**

Deploy the script on the Linux server and use

```
crontab -e
```

to set the schedule.






