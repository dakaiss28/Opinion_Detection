USE master ;
GO
CREATE DATABASE tweets
GO 

DROP TABLE  IF EXISTS dbo.tweets;
GO
-- Create the table in the specified schema
CREATE TABLE dbo.tweets
(
 tweet_id NVARCHAR(50) NOT NULL,
 created_at DATETIME,
 content NVARCHAR(300),
 nb_retweets INT,
 nb_fav INT,
 topic NVARCHAR(50),
 target_label INT,
 kmeans_res INT,
 svm_res INT
);
GO