+++
date = "2016-05-10T01:55:26+02:00"
description = ""
draft = true
tags = ["Reading"]
title = "Some Trading Basics"
topics = []

+++

This week I got an interesting opportunity to get involved in an intensive business camp offered by Flow Traders, an Amsterdam-based trading group. So I read some ABC-materials in advance, then hopefully I can at least say something during the discussion :)  

**Most of these contents are reading notes, NOT my own writing**  

___  


### Flow Traders Business Course 2016

#### Concepts

##### High-frequency trading (HFT)  

High-frequency trading (HFT) is a program trading platform that uses powerful computers to **transact a large number of orders at very fast speeds**. High-frequency trading uses complex algorithms to **analyze multiple markets and execute orders based on market conditions**. Typically, the traders with the fastest execution speeds will be more profitable than traders with slower execution speeds. As of 2009, it is estimated more than 50% of exchange volume comes from high-frequency trading orders. 

High-frequency trading became most popular when exchanges began to offer **incentives** for companies to **add liquidity** to the market. For instance, the New York Stock Exchange has a group of liquidity providers called supplemental liquidly providers (SLPs), which attempt to add competition and liquidity for existing quotes on the exchange. As an incentive to the firm, the NYSE pays a fee or rebate for providing said liquidity. As of 2009, the SLP rebate was $0.0015. Multiply that by millions of transactions per day and you can see where part of the profits for high frequency trading comes from.

The SLP was introduced following the collapse of Lehman Brothers in 2008, when liquidity was a major concern for investors.  

高频率、低延迟  co-location front-run，诱饵订单


有两种策略，做市（market making）和套利（arbitrage），从性价比来说，做市是更好的选择。  

做市是指，在市场上充当流动性提供者，通俗的说就是有任何人想买一个东西（比如股票，期货等），你要保证能卖给他，有任何人想卖一个东西，你要保证从他那买。保证的意思就是如果市场上没有别人出头，做市商就必须出来。承担风险赚取价格波动或对冲（比如买进一只股票的同时卖出它的期货）  

套利是指，找到两种强相关性的证券。一个极端的例子是，ETF和组成ETF的那些股票。如果你知道ETF的计算方式，就可以用同样的方式通过那些股票的价格来计算一个ETF的期望价格。有的时候，因为种种原因，你发现这个价格和你在市场上看到的ETF价格不一样，你就知道显然是市场发生了一些混乱，早晚这个价格会变回来。这时你就可以买入（卖出）ETF，卖出（买入）那些股票，坐等价格回归，可以稳赚不赔。这个策略听起来很美，实际上竞争非常激烈。因为任何人都可以做这件事，参与的人多了，市场就会少犯错误，同时每个人的利润空间也变小了。当你的套利收入不足以支撑HFT的研发维护成本的时候，离关门也就不远了。  

HFT可能存在的问题。Flash crash是真实发生过的，也是最大的隐患。当一个市场上70%的交易都是HFT完成的时候，我们必须要能对HFT的系统有信心。这就需要HFT的开发流程标准化，接受开发过程的评审，有严格的测试体系。几个技术宅关在小黑屋里鼓捣出来的东西没人敢拍胸脯保证不会死机。而这一点目前看的确是比较差的，需要尽快规范起来。这才是公众需要关注的重点。  

##### Algorithmic Trading  


Algorithmic trading is a trading system that utilizes very advanced mathematical models for making transaction decisions in the financial markets. The strict rules built into the model **attempt to determine the optimal time for an order to be placed that will cause the least amount of impact on a stock's price**. Large blocks of shares are usually purchased by dividing the large share block into smaller lots and allowing the complex algorithms to decide when the smaller blocks are to be purchased.  

The use of algorithmic trading is most commonly used by large institutional investors due to the large amount of shares they purchase everyday. Complex algorithms allow these investors to obtain the best possible price without significantly affecting the stock's price and increasing purchasing costs.  

冲击成本 impact cost  

##### Market Maker
A market maker is a broker-dealer firm that accepts the risk of holding a certain number of shares of a particular security in order to facilitate trading in that security. Each market maker competes for customer order flow by displaying buy and sell quotations for a guaranteed number of shares. Once an order is received, the market maker immediately sells from its own inventory or seeks an offsetting order. This process takes place in mere seconds.  

The Nasdaq is the prime example of an operation of market makers. There are more than 500 member firms that act as Nasdaq market makers, keeping the financial markets running efficiently because they are **willing to quote both bid and offer prices for an asset.**  

**"Liquidity provider" is essentially synonymous with "market maker."**  

##### Arbitrage vs. Spreading vs. Speculation  

- Arbitrage:  simultaneous buying and selling; a type of hedge and involves limited risk; typically enter large positions since they are attempting to profit from very small differences in price.  
- Spreading: Futures contracts
- Speculation: Profit from rising and falling prices; involves a significant amount of risk; longing when expect rising, shorting when expect falling.  









