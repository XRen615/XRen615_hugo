+++
author = "X.Ren"
comments = true
date = "2016-03-06T01:27:21+01:00"
draft = false
image = ""
menu = ""
share = true
slug = "develope_this_site"
tags = ["Site maintenance"]
title = "建站記錄"
+++

**Documentation**: how to initialize a basic static site like this with Github page, Hugo and Go.  
**記錄文檔**：如何使用Github page，Hugo和Go便利地建造本站的基礎元素.

### System Enviroment

OSX El Capitan 10.11.3  
MacBook Pro（Retina，13 inch, early 2015)  
CPU: 2.7 GHz Intel Core i5  
Graphics: Intel Iris Graphics 6100 1536 MB

### Steps  

1. Hugo is compiled by Go, thus before everything, install [Go](https://golang.org) and set up its [enviroment path](http://blog.csdn.net/lan2720/article/details/20767941).
2. Follow this [link](http://blog.coderzh.com/2015/08/29/hugo/) to install Hugo and set up everything. You may also need [another link](http://blog.bpcoder.com/2015/12/hugo-create-blog/).  

### Tips:  

- Path may vary with the theme you chose, so always check with the [theme page](https://github.com/spf13/hugoThemes) for what path should be used for post.  
- Always remember: **public** push to [your github account].github.io for publication, (optional) parent dir. push to [your github account]_hugo for archive.
- Change the social icon: Social links are designed with [FontAwesome 4.5.0](https://fortawesome.github.io/Font-Awesome/). For occasional modification, just edit the *header.html*, find:  
> *class="fa fa-xxx"*   
substitute *xxx* with the icon name you want (and ofc supported by FontAwesome), do not touch .Site.Params.xxx if you just want it simple.  
- If you are not familiar with *markdown*, find the instruction regarding markdown syntax [here](http://www.jianshu.com/p/q81RER).  
- Develope comment board with disqus: register a short name then fill it in the config file
- Change the Square to superscript： edit /public/index.html, replace INCH2 with INCH<sup>2</sup> except for the <title> label.


### Paste some frequently-used order here:)  

	hugo new post/first.md
	hugo server -w **or** hugo server --theme=slender --buildDrafts --watch
	hugo **or** hugo --theme=slender --baseUrl="http://xren615.github.io/"
