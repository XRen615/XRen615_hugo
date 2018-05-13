+++
date = "2017-05-21T22:44:24+08:00"
description = ""
draft = false
tags = ["leetcode"]
title = "Leetcode locked questions"
topics = []
slug = "leetcode"

+++

**170 Two Sum III**  

Design and implement a TwoSum class. It should support the following operations:add and find.

add - Add the number to an internal data structure. 
find - Find if there exists any pair of numbers which sum is equal to the value.

For example, 
add(1); add(3); add(5); 
find(4) -> true 
find(7) -> false



```
class TwoSum:
    
    def __init__(self):
        self.table = {}
    
    def add(self,num):
    	  # store how many times each num appears
        self.table[num] = self.table.get(num, 0) + 1
        
    def find(self, target):
        for i in self.table.keys():
            j = target - i
        if i==j and self.table[i] > 1 or i!=j and self.table[j] > 0:
            return True
        return False  
```

**243 SHORTEST WORD DISTANCE**

Given a list of words and two words word1 and word2, return the shortest distance between these two words in the list.

For example,
Assume that words = ["practice", "makes", "perfect", "coding", "makes"].

Given word1 = “coding”, word2 = “practice”, return 3.
Given word1 = "makes", word2 = "coding", return 1.

Note:
You may assume that word1 does not equal to word2, and word1 and word2 are both in the list.


Hints:
Two variable to track the last positions.

```  
class Solution:
    # @param {string[]} words
    # @param {string} word1
    # @param {string} word2
    # @return {integer}
    def shortestDistance(self, words, word1, word2):
        idx1 = [i for i in range(len(words)) if words[i] == word1]
        idx2 = [i for i in range(len(words)) if words[i] == word2]
        return min([abs(i - k) for i in idx1 for k in idx2])    
        
            
```  

**244 SHORTEST WORD DISTANCE II**  

This is a follow up of Shortest Word Distance. The only difference is now you are given the list of words and your method will be called repeatedly many times with different parameters. How would you optimize it?  

Hint: 这次我们需要多次调用求最短单词距离的函数，用哈希表来建立每个单词和其所有出现的位置的映射，然后在找最短单词距离时，我们只需要取出该单词在哈希表中映射的位置数组进行两两比较即可

```
class WordDistance(object):
	def __init__(self, words):
	# initialize a dict with value as list
        self.d = collections.defaultdict(list)
        for i, w in enumerate(words):
            self.d[w].append(i)
	def shortest(self, word1, word2):
		return min([abs(n1 - n2) for n1 in self.d[word1] for n2 in self.d[word2]])
```  

**245 SHORTEST WORD DISTANCE III** 

This is a follow up of Shortest Word Distance. The only difference is now word1 could be the same as word2.  

```
class WordDistance(object):
	def __init__(self, words):
	# initialize a dict with value as list
        self.d = collections.defaultdict(list)
        for i, w in enumerate(words):
            self.d[w].append(i)
	def shortest(self, word1, word2):
		return min([abs(n1 - n2) for n1 in self.d[word1] for n2 in self.d[word2] if n1!=n2])
```  

**580. Count Student Number in Departments**

```
SELECT d.dept_name, t.student_number FROM
department d 
LEFT JOIN
(SELECT dept_id, count (student_id) AS student_number FROM
student s 
GROUP BY dept_id) t ON d.dept_id = t.dept_id
ORDER BY student_number DESC, dept_name DESC
```  

**K closest points**

Find the K closest points to the origin in a 2D plane, given an array containing N points.  

```
def kClosestPoints(points):  
	distances = {}
	for i, point in enumerate(points):
		distances[i] = points[0]**2 + points[1]**2
	#sort the hash by convert to tuple list
	return sorted(distances.items(),key = lambda x:x[1])[:k-1]
```  

**578. Get Highest Answer Rate Question**  

```  
SELECT t.question_id AS survey_log FROM 
(SELECT question_id, SUM(CASE WHEN action ='show' THEN 1 ELSE 0 END) AS show_num, SUM(CASE WHEN action = 'answer' THEN 1 ELSE 0 END) AS answer_num 
FROM survey_log
GROUP BY question_id) t
ORDER BY answe_num/show_num DESC LIMIT 1
```  

**585 Investments in 2016**  

```
SELECT
    SUM(insurance.TIV_2016) AS TIV_2016
FROM
    insurance
WHERE
    insurance.TIV_2015 IN
    (
      SELECT
        TIV_2015
      FROM
        insurance
      GROUP BY TIV_2015
      HAVING COUNT(*) > 1
    )
    AND CONCAT(LAT, LON) IN
    (
      SELECT
        CONCAT(LAT, LON)
      FROM
        insurance
      GROUP BY LAT , LON
      HAVING COUNT(*) = 1
    )
;
```

**277 Find celebrity**

Suppose you are at a party with n people (labeled from 0 to n - 1) and among them, there may exist one celebrity. The definition of a celebrity is that all the other n - 1people know him/her but he/she does not know any of them.

Now you want to find out who the celebrity is or verify that there is not one. The only thing you are allowed to do is to ask questions like: "Hi, A. Do you know B?" to get information of whether A knows B. You need to find out the celebrity (or verify there is not one) by asking as few questions as possible (in the asymptotic sense).

You are given a helper function bool knows(a, b) which tells you whether A knows B. Implement a function int findCelebrity(n), your function should minimize the number of calls to knows.

Note: There will be exactly one celebrity if he/she is in the party. Return the celebrity's label if there is a celebrity in the party. If there is no celebrity, return -1.

```
def findCelebrity(self, n):
    x = 0
    for i in xrange(n):
        if knows(x, i):
            x = i
    if any(knows(x, i) for i in xrange(x)):
        return -1
    if any(not knows(i, x) for i in xrange(n)):
        return -1
    return x
    
```  

Explanation

The first loop is to exclude n - 1 labels that are not possible to be a celebrity.
After the first loop, x is the only candidate.
The second and third loop is to verify x is actually a celebrity by definition.

The key part is the first loop. To understand this you can think the knows(a,b) as a a < b comparison, if a knows b then a < b, if a does not know b, a > b. Then if there is a celebrity, he/she must be the "maximum" of the n people.

However, the "maximum" may not be the celebrity in the case of no celebrity at all. Thus we need the second and third loop to check if x is actually celebrity by definition.

The total calls of knows is thus 3n at most. One small improvement is that in the second loop we only need to check i in the range [0, x). You can figure that out yourself easily.



**256 Paint Houses**

There are a row of n houses, each house can be painted with one of the three colors: red, blue or green. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by a n x 3 cost matrix. For example,costs[0][0] is the cost of painting house 0 with color red; costs[1][2] is the cost of painting house 1 with color green, and so on... Find the minimum cost to paint all houses.

Note:
All costs are positive integers.

```
# dynamic programming case
from collections import defaultdict
def minCost(costs):
	dp = defaultdict(list)
	#dp[i][j] is the min cost if we paint house i with color j
	dp[0][0] = costs[0][0]
	dp[0][1] = costs[0][1]
	dp[0][2] = costs[0][2]
	for i in range(1,n+1):
		dp[i][0] = min(dp[i-1][1], dp[i-1][2]) + costs[i][0]
		dp[i][1] = min(dp[i-1][0], dp[i-1][2]) + costs[i][1]
		dp[i][2] = min(dp[i-1][0], dp[i-1][1]) + costs[i][2]
	return min(dp[n])
```  

**254 Factor combination** 

Numbers can be regarded as product of its factors. For example,

8 = 2 x 2 x 2;
  = 2 x 4.
Write a function that takes an integer n and return all possible combinations of its factors.

Note: 
Each combination's factors must be sorted ascending, for example: The factors of 2 and 6 is [2, 6], not [6, 2].
You may assume that n is always positive.
Factors should be greater than 1 and less than n  

思路：DFS + backtracking。每次搜索从最小的一个因子开始到sqrt(n)查看之间的数是否可以被n整除，如果可以那么有两种选择：

1. n 除以这个因子，并保持这个因子， 然后进行下一层搜索。

2. 不再进行下一层搜索，保存当前结果。

其中 factor > sqrt(n) 是一个关键的剪枝, 保证了没有重复解.

```
# DFS
def factor(n,starter):
    ans = []
    for i in range(starter,int(n**0.5)+1):
        if n%i == 0:
            ans.append([i,n/i])
            # starter is designed to avoid duplication
            for f in factor(n/i,i):
                ans.append([i] + f)
    return ans
```  


**156 Binary Tree Upside Down**  

```
class Solution:
        # @param root, a tree node
        # @return root of the upside down tree
        def upsideDownBinaryTree(self, root):
            # take care of the empty case
            if not root:
            	return root
            	
            l = root.left
            r = root.right
            root.left = None
            root.right = None
            
            while l:
            	newL = l.left
            	newR = l.right
            	l.left = r
            	l.right = root
            	l = newL
            	r = newR
            	root = l
            
            return root      
```  

**311 Sparse Matrix Multiplication**  

Given two sparse matrices A and B, return the result of AB.

You may assume that A's column number is equal to B's row number.  

```
class Solution(object):
    def multiply(self, A, B):
        """
        :type A: List[List[int]]
        :type B: List[List[int]]
        :rtype: List[List[int]]
        """
        if A is None or B is None:
        	return None
        m,l = len(A), len(B[0])
        #initialize a 0 matrix, change non-zero positions later.
        C = [[0 for _ in range(l)] for _ in range(m)]
        for i, row in enumerate(A):
        	for k, eleA in enumerate(row):
        		if eleA:
        			for j, eleB in enumerate(B[k])# C[i][j] = Sigmak(A[i][k]*B[k][j])
        				if eleB:
        					C[i][j] +=eleA * eleB
        return C
```

 