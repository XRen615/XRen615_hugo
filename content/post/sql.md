+++
author = "X.REN"
comments = true
date = "2016-04-01T18:49:28+02:00"
draft = false
image = ""
menu = ""
share = true
slug = "sql"
tags = ["SQL", "Note"]
title = "SQL Notes"

+++

(This is the note for the SQL training course on Codecademy.com)  
#### Some basic recipes  

	SELECT DISTINCT  
	
specifies that the statement is going to be a query that returns unique values in the specified column(s)  

**WHERE**  

	SELECT * FROM movies
	WHERE year BETWEEN 1990 AND 2000
	AND genre = 'comedy'; 
	
**LIKE**: a special operator used with the WHERE clause to search for a specific pattern in a column.  
**%**: wildcard character, matches zero or more missing letters in the pattern.

	SELECT * FROM movies WHERE name like 'a%';  
	
**ORDER BY**: a clause that indicates you want to sort the result set by a particular column either alphabetically or numerically.  

	SELECT * FROM movies
	ORDER BY imdb_rating ASC/DESC;  
	
**LIMIT**: a clause that lets you specify the maximum number of rows the result set will have.  

	SELECT * FROM Persons
	LIMIT 5;  
	
#### Using aggregate functions to perform calculations  

Aggregate functions combine multiple rows together to form a single value of more meaningful information.  

**COUNT()**: a function that takes the name of a column as an argument and counts the number of rows where the column is not NULL. Here, we want to count every row so we pass * as an argument.  

	SELECT COUNT(*) FROM fake_apps;  
	
**GROUP BY**: Count the number of apps at each price. 

	SELECT price, COUNT(*) FROM fake_apps
	GROUP BY price;  
	
**SUM()**: a function that takes the name of a column as an argument and returns the sum of all the values in that column.  

	SELECT SUM(downloads) FROM fake_apps;  
	
**MAX()/MIN()**: a function that takes the name of a column as an argument and returns the largest/smallest value in that column.  


	SELECT name, category, MAX(downloads) FROM fake_apps
	GROUP BY category;  
	
**AVG()**: a function works by taking a column name as an argument and returns the average value for that column.  

**ROUND()**: a function that takes a column name and an integer as an argument. It rounds the values in the column to the number of decimal places specified by the integer.  

#### Query multiple tables that have relationships with each other  

**primary key**  

A primary key serves as a unique identifier for each row or record in a given table. The primary key is literally an id value for a record. We're going to use this value to connect artists to the albums they have produced.  

	CREATE TABLE 
	artists(id INTEGER PRIMARY KEY, name TEXT);

By specifying that the id column is the PRIMARY KEY, SQL makes sure that:  

- None of the values in this column are NULL  
- Each value in this column is unique  

A table can not have more than one PRIMARY KEY column.

**foreign key**

A foreign key is a column that contains the primary key of another table in the database. We use foreign keys and primary keys to connect rows in two different tables. One table's foreign key holds the value of another table's primary key. Unlike primary keys, foreign keys do not need to be unique and can be NULL.  

**JOIN**: An inner join will combine rows from different tables if the join condition is true.  

	SELECT * FROM albums
	JOIN artists ON
	albums.artist_id = artists.id;  
	
**LEFT JOIN**: every row in the left table （the former） is returned in the result set, and if the join condition is not met, then NULL values are used to fill in the columns from the right table.  

	SELECT * FROM albums
	LEFT JOIN artists ON
	albums.artist_id = artists.id;

**AS**: a keyword in SQL that allows you to rename a column or table using an alias.  

	SELECT 
	albums.name AS 'Album',
    albums.year,
    artists.name AS 'Artist'
    FROM
    albums
    JOIN artists ON
    albums.artist_id = artists.id
    WHERE
    albums.year > 1980;



	




