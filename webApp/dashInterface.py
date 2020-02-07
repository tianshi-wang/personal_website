"""SQL queries for webapp
The app.py gets the needed pandas dataframes via this module.
This module connects PostgreSQL databases on AWS RDS.
Return: aggregate data for webapp display
"""

import pandas as pd
from sqlalchemy import create_engine


def conn_aws_rds():
    """
    Connect to temporary AWS RDS server for test purpose only
    :return: SQL engine
    """
    dbname = 'birth_db'  # DB name not table
    # Username for test usage only
    username = 'Vera'
    # Password for test usage only
    passwd = '111111aa'
    # AWS RDS for test usage only
    hostAddr = 'insightdb.c4f4cvkgxat9.us-east-2.rds.amazonaws.com:5432'
    awsEngine = create_engine('postgresql+psycopg2://%s:%s@%s/%s' % (username, passwd, hostAddr, dbname))
    return awsEngine


def inventoryLevel():
    """
    :return: format as "moduleName, inventory"
    """
    engine = conn_aws_rds()
    sql_query = """
    SELECT * FROM inventorylevel Limit 10    ;
    """
    df = pd.read_sql_query(sql_query, engine)
    df = df.drop(['numinv', 'numwant'], axis=1)
    df = df.fillna(value=0)
    return df


def userTable(categories):
    """
    Return a table to display on website
    :param categories: the categories selected by the user on website
    :return:
    Row: top 200 collectors ordered by score DESC
    Columns: Probability to sell, the percentage of each category in the user's whole collection
    and final score.
    """
    if len(categories) == 1:
        categories = (categories[0], '')
    engine = conn_aws_rds()
    sql_query = """
    with topusers as (
        select t1."userId", t1."weight", t2."likelihood", t1."weight"*t2."likelihood"*100 as score
        from(
            select "userId", sum("count")/avg("totalcoll") as weight
            from collectionbyuser
            where "ModuleName" in {}
            group by "userId"
            ) as t1
        join likelihood as t2 
        on t1."userId"=t2."userId"
        order by score DESC
        limit 100
    )

    select collectionbyuser.*, t2.weight, t2."likelihood",t2.score
    from collectionbyuser
    left join topusers as t2
    on collectionbyuser."userId"=t2."userId"
    where collectionbyuser."ModuleName" in {} and t2.score is not Null
    order by t2.score DESC
    limit 60
    """.format(tuple(categories), tuple(categories))
    df = pd.read_sql_query(sql_query, engine, index_col='userId')
    df['percent'] = df['count'] / df['totalcoll']
    df_left = df.pivot(columns='ModuleName', values='percent')
    df_left = df_left.fillna(value=0)
    result = df_left.join(df[['likelihood', 'score']]).sort_values(by='score', ascending=False)
    for column in result.columns:
        result[column] = result[column].map(lambda n: '{:.2f}'.format(n))
    result = result.reset_index()
    return result


def top3lowest():
    """
    :return: a list of three categories e.g. ['funko', ...]
    """
    engine = conn_aws_rds()
    sql_query = """
    SELECT * FROM inventorylevel Limit 10
    """
    df = pd.read_sql_query(sql_query, engine)
    column = df.columns[-1]
    df = df.fillna(value=0)
    df = df.sort_values(by=column)
    return list(df.module[0:3])


def wishlistGroupbyModule():
    """
    :return: Format
    1. Row: module names
    2. Columns: new wishlist by month in the past 12 month
    """
    engine = conn_aws_rds()
    sql_query = """
    SELECT * FROM wishlistgroupbycategory Limit 5    ;
    """
    df = pd.read_sql_query(sql_query, engine)
    return df


def ordersGroupbyCategory():
    """
    :return: format as "userId, moduleName, year, month, count"
    """
    engine = conn_aws_rds()
    sql_query = """
    SELECT * FROM ordersgroupbycategory Limit 5    ;
    """
    df = pd.read_sql_query(sql_query, engine)
    return df.iloc[:,:-1]

def users():
    """
    :return: format as "userId, moduleName, year, month, count"
    """
    engine = conn_aws_rds()
    sql_query = """
    SELECT * FROM users   ;
    """
    df = pd.read_sql_query(sql_query, engine)

    return df

def usercoll(categories=("""SELECT DISTINCT "ModuleName" FROM collectiongroupbyuserandmodule""")):
    engine = conn_aws_rds()
    sql_query = """
    SELECT *
    FROM collectiongroupbyuserandmodule
    WHERE "ModuleName" IN {:s}
    """.format(categories)
    df = pd.read_sql_query(sql_query, engine)
    return df


def collectionGroupby():
    engine = conn_aws_rds()
    """
    Read collections table and write new table collectiongroupby
    :return: format as "userId, moduleName, year, month, count"
    """
    # query:
    sql_query = """
    SELECT * FROM collectiongroupbymodule Limit 5
    ;
    """
    df = pd.read_sql_query(sql_query, engine)
    return df


def summary():
    """
    :return: dataframe in the format:
    1. Rows: NumberOfOrders, NumberOfCollections, NumberOfWishlist, NumberOfUsers,NumberOfSellers
    2. Columns: the past 12 months, 2017-01 as 1
    """
    engine = conn_aws_rds()
    sql_query = """
    SELECT * FROM summary ;
    """
    df = pd.read_sql_query(sql_query, engine)
    # Does not return the last column (sum) of SQL table
    return df.iloc[:,:-1]


def collectionGroupbyModule():
    """
    :return: format as "userId, moduleName, year, month, count"
    """
    engine = conn_aws_rds()
    sql_query = """
    SELECT * FROM collectiongroupbymodule Limit 5    ;
    """
    df = pd.read_sql_query(sql_query, engine)
    return df



