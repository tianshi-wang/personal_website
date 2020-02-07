# Buy2Sell
### Converting collectors to buyers for Covetly, an online marketplace for collectible toys

# Table of content
1. [Problem](README.md#problem)
2. [Approach](README.md#approach)
3. [Technical Architecture](README.md#technical-architecture)
4. [Code Structure](README.md#code-structure)
5. [Module Description](REAME.md#module-description)
6. [Contacts](README.md#contacts)

# Problem
Covetly suffers from the high variance of inventory levels across categories. To add inventory for categories
 in shortage, the client expects this project to suggest a list of collectors who can be prompted to sell. 
 However, simply prompting collectors with related inventories is not good enough for two reasons. Firstly, the converted sellers 
 should be able to sell actively because the platform actually may lose money if the seller only complete one or two
 orders per year due to management cost. Secondly, the seller's inventory composition should match the company's need
 very well. It is not so good if a seller tries to sell things the platform already has much inventory.
 
 
# Approach
This project provide an approach to increase inventory levels by converting collectors to sellers. The model pipeline
comprises of three components. Firstly, the model use classification method to find the similarity between consumption
behaviors of sellers who sold multiple orders in the past. The model is then used to predict the probability of selling
for each collector. The second step provides a percentage by matching inventory composition to categories in shortage. 
At the end, the final score is the multiplication of the probability and percentage. 

# Technical Architecture
The data is from the client's MongoDB and MSSQL databases. A data warehouse is created on AWS RDS which can be updated 
daily. On the data warehouse, about 12 tables are created containing aggregate data for model and webapp. The model and 
webapp are hosted on AWS EC2. 

# Code Structure

    ├── README.md 
    ├── run.sh
    ├── src
    │   └── cache.py
    │   └── connLocalDB.py
    │   └── connMongo.py
    |   └── downloadFromCovetly.py
    │   └── modelTraining.py
    │   └── syncAwsRDS.py
    │ 
    ├── webapp
    │   └── app.py
    │   └── controls.py
    │   └── dashInterface.py
    │
    ├── data
        └── logo.png

# Module Description
A brief introduction for the function of key modules. For details, please read the docstrings.</br>

Under src folder:</br>
- cache.py: Query SQL data and store aggregate data as ~15 tables on local DB.
- connLocalDB.py: Connect to local DB.
- connMongo.py: Connect to client's MongoDB.
- downloadFromCovetly.py: Download needed data from MongoDB to local DB.
- modelTraining.py: Train classification model and predict the probability to sell for each collector.
- syncAwsRDS.py: Sync some tables on local DB to AWS RDS for webapp and model training.

Under webapp folder:</br>
- app.py: Main code for webapp display
- controls.py: Internal dictionaries and lists
- dashInterface.py: SQL query for webapp

# Contact
The webapp is on www.tianshi-wang.com </br>
Feel free to contact me if you have any question or comment.

Tianshi Wang </br>
tianshi_wang@outlook.com </br>
https://www.linkedin.com/in/tianshi-wang/
