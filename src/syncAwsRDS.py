"""Copy tabels from local Postgresql to AWS RDS Postgresql
Local DB name: insightProj
AWS Instance Name: insightdb, AWS DB name: birth_db
"""

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd


def connAWS():
    # Temporary AWS RDS database for test purpose only
    dbname = 'birth_db'
    # Input username and passwd
    username = ''  
    passwd = ''
    hostAddr = 'insightdb.c4f4cvkgxat9.us-east-2.rds.amazonaws.com:5432'
    awsEngine = create_engine('postgresql+psycopg2://%s:%s@%s/%s' % (username, passwd, hostAddr, dbname))
    if not database_exists(awsEngine.url):
        create_database(awsEngine.url)
    return awsEngine


class LocalToAWS:
    def __init__(self):
        self.df = None
        self.dfList = None
        self.tableName = None
        self.awsEngine = None
        self.localEngine = None
        self.awsEngine = connAWS()

    def connLocalDB(self):
        """
        Connect to local standby DB
        """
        dbname = 'insightProj'
        username = 'Vera'  # change this to your username
        self.localEngine = create_engine('postgres://%s@localhost/%s' % (username, dbname))

    def migrateTables(self, tableToMove):
        """
        Copy tables in local DB to AWS RDS
        :param tableToMove: A list of tables to be copied
        """
        def migrateOneTable(table):
            """
            Copy one table by table name
            """
            sql_query = "SELECT * FROM %s " % (table)
            df_table = pd.read_sql_query(sql_query, con=self.localEngine)
            df_table.to_sql(name=table, con=self.awsEngine, if_exists='replace', index=False)
            print("Finished " + table + "...")

        if type(tableToMove) == str:
            migrateOneTable(tableToMove)
        else:
            for table in tableToMove:
                migrateOneTable(table)


def main():
    """
    Copy selected tables stored on local PostgreSQL to AWS RDS
    The selected tables are listed in "tabelToMove"
    """
    writeAWS = LocalToAWS()
    writeAWS.connLocalDB()
    print("Table names on local DB (self.tableNameLocal to access):")
    writeAWS.tableNameLocal = writeAWS.localEngine.table_names()
    print(writeAWS.tableNameLocal)

    # Print table names on AWS DB
    print("Table names on AWS DB (self.tableNameAWS to access):")
    writeAWS.tableNameAWS = writeAWS.awsEngine.table_names()
    print(writeAWS.tableNameAWS)

    tableToMove = ['wishlistgroupbycategory',
                   'likelihood',
                   'collectionbyuser',
                   'orders',
                   'ordersgroupbycategory',
                   'ordersgroupbyusersnum',
                   'ordersgroupbyusersamount',
                   'collectiongroupbymodule',
                   'collectiongroupby',
                   'summary',
                   'collectiongroupbyuserandmodule',
                   'inventorylevel',
                   ]
    writeAWS.migrateTables(tableToMove)


if __name__ == "__main__":
    main()

