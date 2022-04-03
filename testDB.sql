PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE ImageTable (id int primary key, label varchar(30), img_feature text);
CREATE TABLE SignatureTable (id int primary key, label varchar(30), sig_feature text);
CREATE TABLE WristTable (id int primary key, label varchar(30), wri_feature text);
COMMIT;
