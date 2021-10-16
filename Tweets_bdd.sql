USE master ;
GO
CREATE DATABASE tweets
GO 

DROP TABLE  IF EXISTS dbo.tweets;
GO
-- Create the table in the specified schema
CREATE TABLE dbo.tweets
(
 tweet_id BIGINT NOT NULL,
 created_at DATETIME,
 content NVARCHAR(300),
 nb_retweets INT,
 nb_fav INT,
 brand NVARCHAR(50),
 label INT
);
GO