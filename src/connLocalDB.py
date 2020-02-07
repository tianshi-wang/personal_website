"""Connect to LocalDB

"""
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd


def connDB():
    """
    The local DB is for test purpose only.
    :return: engine and conn for SQL query
    """
    dbname = 'insightProj'
    username = 'Vera'  # change this to your username
    engine = create_engine('postgres://%s@localhost/%s' % (username, dbname))
    if not database_exists(engine.url):
        create_database(engine.url)

    conn = psycopg2.connect(database=dbname, user=username)
    return engine, conn


def runQuery(sql_query):
    """
    Read collections table and write new table collectiongroupby
    :return: format as "userId, moduleName, year, month, count"
    """
    conn = connDB()
    df = pd.read_sql_query(sql_query, conn)
    return df