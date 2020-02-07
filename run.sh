#/bin/bash

#Grep needed data from client's MongoDB and msSQL databases
#pw are not provided with this repository
#The output is a local PostgreSQL database: InsightProj
python3  ./src/pipelineToSQL.py  pw_for_invDB pw_for_domainDB 

#Do SQL query and write ~10 tables on the local database
#e.g. OrderByUser, CollectionByUserAndCategory, etc..
python3 ./src/cache.py

#Train Machine Learning model to find the high-impact sellers
python3 ./src/modelTraining.py

#Copy some tables to AWS RDS for online dashboard
python3 ./src/syncAwsRDS.py  'repalceWithMyPassword'

#Test the local website
python3 ./app.py
