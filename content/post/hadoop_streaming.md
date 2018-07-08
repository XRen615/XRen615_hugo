+++
draft = false
tags = ["hadoop"]
date = "2018-07-08T20:58:01+08:00"
author = "X.Ren"
comments = true
share = true
slug = "hadoop_streaming"
title = "Use Python on HDFS with Pig Latin UDF and Hadoop streaming"
+++

在大多数情景中，当需要对HDFS上的数据做一些简单的ETL，我们常常直接选择Hive或者Apache Pig Latin来完成。而在其他少数情况下，如果想要插入其他脚本语言模块，如Python，来完成一些比较复杂的工作，这时我们一般有两种选择，UDF (User Defined Function) 或者Hadoop Streaming。

**UDF with Python**

It's simple to use Py UDF with pig, just put a .py file and a .pig file under the same directory.   

E.g.

udf_testing.py  

```
@outputSchema('word:chararray')
def hi_world():
    return "hello world”
def bingo(s):
    return s + 'bingo'
```

udf_testing.pig  

```
REGISTER 'udf_testing.py' using jython as my_udfs;
page_views = LOAD '/data/tracking/PageViewEvent/' USING LiAvroStorage('date.range', 'start.date=20171001;end.date=20171002;error.on.missing=false');
hello_users = FOREACH page_views GENERATE requestHeader.pageKey, my_udfs.hi_world(), my_udfs.bingo(requestHeader.pageKey);;
DUMP hello_users;
```  

Note: limitation from Jython

There are a couple of gotchas to using Python in the form of a UDF. Firstly, even though we are writing our UDFs in Python, Pig executes them in Jython. Jython is an implementation of Python that runs on the Java Virtual Machine (JVM). Most of the time this is not an issue as Jython strives to implement all of the same features of CPython but there are some libraries that it doesn't allow. For example you can't use numpy from Jython.

**Working with Hadoop Streaming**

Hadoop allows you to write mappers and reducers in any language that gives you access to stdin and stdout. The aim is to get Hadoop to run the script on each node. This allows us to get around the Jython issue when we need to.

Example 1  

streaming_test.py  

```
#!/usr/bin/python
import sys
import string
for line in sys.stdin:
    print str(line)[:5]
```  

To make the streaming UDF accessible to Pig we make use of the define statement. 

```
DEFINE alias 'command' SHIP('files');
```  

The alias is the name we use to access our streaming function from within our PigLatin script. The command is the system command Pig will call when it needs to use our streaming function. And finally SHIP tells Pig which files and dependencies Pig needs to distribute to the Hadoop nodes for the command to be able to work.

streaming_test.pig  

```
DEFINE CMD `streaming_test.py` SHIP ('/export/home/riren/ad_hoc/streaming_test.py');
feedViews = LOAD '/data/tracking/FeedActionEvent/' USING LiAvroStorage('date.range','start.date=20180401;end.date=20180402;error.on.missing=false');
streamed = STREAM feedViews THROUGH CMD; 
DUMP streamed;  
```

Example 2

以下数据为一部分儿童的信息数据sampleTest.csv，第一个属性为用户id,birthday为用户的生日，gender表示男女，0为女，1为男，2为未知。假设我们的问题是：在这些儿童中，每一年出生的男孩和女孩各是多少。  
  
```
user_id,birthday,gender
2757,20130311,1
415971,20121111,0
1372572,20120130,1
10339332,20110910,0  
```
参考mapreduce的工作流程，我们可以直接使用Python操作标准输入输出编写mapper和reducer  

![mapreduce](http://7xro3y.com1.z0.glb.clouddn.com/mapreduce.jpg)  

Mapper  

```
import sys

for data in sys.stdin:
    data = data.strip()
    record = data.split(',')
    user_id = record[0]
    if user_id == "user_id":
            continue
    birthyear = record[1][0:4]
    gender = record[2]
    sys.stdout.write("%s\t%s\n"%(birthyear,gender))
```  

Reducer  

```
import sys

numByGender = {'0':0,'1':0,'2':0}
lastKey = False
for data in sys.stdin:
    data = data.strip()    
    record = data.split('\t')
    curKey = record[0]
    gender = record[1]
    if lastKey and curKey !=lastKey:
        sys.stdout.write("%s year:%s female,%s male \n"%(lastKey,numByGender['0'],numByGender['1']))
        lastKey = curKey
        numByGender = {'0':0,'1':0,'2':0}
        numByGender[gender] +=1
    else:
        lastKey = curKey
        numByGender[gender] += 1
if lastKey:
    sys.stdout.write("%s year:%s female,%s male \n"%(lastKey,numByGender['0'],numByGender['1']))  
```  

管道测试  

```
cat sample.txt | python mapper.py | sort -t ' ' -k 1 | python reducer.py
```  

Hadoop streaming 实测  

```
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.7.4.74.jar \
-D stream.non.zero.exit.is.failure=false \
-files /export/home/riren/experiment/mapper.py,/export/home/riren/experiment/reducer.py \
-input /user/riren/test/sampleTest.csv \
-output /user/riren/test/output \
-mapper "python mapper.py" \
-reducer "python reducer.py"  
```


