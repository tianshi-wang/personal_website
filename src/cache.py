"""
Query SQL data and store aggregate data on local DB.
Write SQL tables:
    OrderGroupby
    CollectionGroupbyUserAndModule
    CollectionByUser
    Summary
    Inventory
    wanttoinv
    WishlistGroupby
    CollectionGroupbyUserAndModule
    Features
    CollectionGroupbyModule
"""
from datetime import datetime

import pandas as pd

from connLocalDB import connDB


def groupby_pivot(df,rowIdx, colIdx, val):
    """
    Process a dateframe like:
    user Month Item
    John Dec   item1
    John Dec   item2
    Andy Jan   item2
    =========================
    To a DF like:
    user Jan ... Dec
    john  0  ... 2
    Andy  1  ... 0
    Params: row index (rowIdx), columns (colIdx), value is count(val) for a given rowIdx and colIdx
    """
    df_series = df.groupby([rowIdx,colIdx])[val].count()
    df_groupby = df_series.to_frame()
    df_groupby.reset_index(inplace=True)
    df_groupby_pivoted = df_groupby.pivot(index=rowIdx, columns=colIdx, values=val).fillna(value=0)
    return df_groupby_pivoted


def groupby(df,first_col, second_col, val):
    """
    Process a dateframe like:
    user Month Item
    John Dec   item1
    John Dec   item2
    Andy Jan   item2
    =========================
    To a DF like:
    John Dec 2
    Andy Jan 1
    ...
    Params: row index (rowIdx), columns (colIdx), value is count(val) for a given rowIdx and colIdx
    """
    df_series = df.groupby([first_col,second_col])[val].count()
    df_groupby = df_series.to_frame()
    return df_groupby


def handle_date(df, col='created_date'):
    df['month'] = df[col].apply(lambda x: (x.year - 2017) * 12 + x.month)  # "added month index 2017-01 as 1"
    df = df.drop([col], axis=1)
    return df

def userClean():
    """
    Clean user table by converting "createdDate" to "month"
    """
    engine, conn = connDB()
    sql_query = """
    SELECT * FROM users
    ;
    """
    df = pd.read_sql_query(sql_query, conn)
    print(df.head())
    df['month'] = df['CreatedDate'].apply(lambda x: (x.year-2017)*12+x.month)
    df = df.drop(['CreatedDate'], axis=1)
    df.to_sql('users', engine, if_exists='replace')


def writeCollectionByUser():
    """
    The format of output:
    user   module   count total_collections
    user1   funko    10     40
    user1   amiibo   15     40
    ...
    """
    engine, conn = connDB()
    sql_query = """
      WITH categorytable AS (
      SELECT collections."userId", items."ModuleName", COUNT(collections."itemId")
      FROM collections
      JOIN items ON collections."itemId" = items."itemId"
      WHERE collections.created_date > '%s'::date-90
      GROUP BY collections."userId", items."ModuleName")
      
      SELECT categorytable.*, sumtable.totalcoll
      FROM categorytable
      JOIN (SELECT "userId", SUM(count) AS totalcoll FROM categorytable GROUP BY "userId") AS sumtable
      ON categorytable."userId"=sumtable."userId"
      
      """ % str(datetime.now().date())
    df_users = pd.read_sql_query(sql_query, conn)
    df_users.to_sql('collectionbyuser', engine, if_exists='replace', index=False)


def writeInventory():
    """
    Output in the following format:
    item    module      category    added_month
    item1   funko       asian       8
    item2   ...
    """
    sql_query="""
    SELECT  "ModuleName", "CategoryName","month", count("itemId")
    FROM inventory
    WHERE "month" >= 9   
    GROUP BY "ModuleName", "CategoryName", "month"
    ORDER BY "ModuleName", "CategoryName", "month" DESC
    """
    engine, conn = connDB()
    df = pd.read_sql_query(sql_query,conn)
    df = handle_date(df, col='CreatedDate')
    df.to_sql('inventorybymodule', engine, if_exists='replace')


def write_wanttoinv():
    """
      Write three tables: order_groupby_userId, order_groupby_category
      (userId and catergary as index, month as columns)
      """
    engine, conn = connDB()
    currMonth = pd.read_sql_query("SELECT MAX(month) FROM inventory", conn).iloc[0,0]
    def query(month):
        return """
          SELECT want.module, inv.numinv, want.numwant, 1.0*inv.numinv/want.numwant AS "{:d}"
          FROM (SELECT module, count("userId") as numwant
          FROM wantlist 
          WHERE month = {:d}-1
          GROUP BY  module) AS want
          LEFT JOIN (SELECT "ModuleName", count("itemId") AS numinv FROM inventory 
          WHERE month>={:d}-3
          GROUP BY "ModuleName"
          ) AS inv
          ON inv."ModuleName"=want.module 
          ORDER BY numwant DESC
          ;
          """.format(month, month, month)
    df = pd.read_sql_query(query(currMonth), conn)
    for month in range(currMonth-1, currMonth-7, -1):
        df_newmonth = pd.read_sql_query(query(month), conn)
        df.insert(loc=3, column=month ,value=df_newmonth.iloc[:,-1])
    df.to_sql("inventorylevel", engine, if_exists='replace')


def writeWishlistGroupby():
    """
      Write three tables: order_groupby_userId, order_groupby_category
      (userId and catergary as index, month as columns)
      """
    engine, conn = connDB()
    sql_query = """
      SELECT want.module, want.numwant, inv.numinv, want.numwant/inv.numinv AS wanttoinv
      FROM (SELECT  module, count("userId") as numwant
      FROM wantlist 
      WHERE month >= (SELECT MAX("month") FROM wantlist AS "maxMonth")-1
      GROUP BY module) AS want
      LEFT JOIN (SELECT "ModuleName", count("itemId") AS numinv FROM inventory 
      WHERE "month">=(SELECT MAX(month) FROM inventory)-3
      GROUP BY "ModuleName"
      ) AS inv
      ON inv."ModuleName"=want.module 
      ;
      """
    df = pd.read_sql_query(sql_query, conn)
    df['month'] = df['created_date'].apply(lambda x: (x.year - 2017) * 12 + x.month)  # "added month index 2017-01 as 1"
    new_df = df.drop(['created_date'], axis=1)
    new_df.to_sql('wishlist1', engine, if_exists='replace')


def writeOrderGroupby():
    """
    Write three tables: order_groupby_userId, order_groupby_category
    (userId and catergary as index, month as columns)
    """
    engine,conn = connDB()
    sql_query = """
    SELECT orders.*, items."CategoryName" FROM orders
    LEFT JOIN items
    ON orders.item_id=items."itemId";
    """
    df = pd.read_sql_query(sql_query, conn)

    # Clean, groupy by and pivot the table
    df['month'] = df['created_date'].apply(lambda x: (x.year-2017)*12+x.month) #"added month index 2017-01 as 1"
    df = df.drop(['created_date'], axis=1)
    df_series = df.groupby(['CategoryName','month'])['order_id'].count()
    df_groupbyCategory = df_series.to_frame()
    df_groupbyCategory.reset_index(inplace=True)
    df_groupbyCategory = df_groupbyCategory.rename(columns={'order_id': 'numOrders'})
    df_groupbyCategory_pivoted = df_groupbyCategory.pivot(index='CategoryName',columns='month',values='numOrders').fillna(value=0)
    df_groupbyCategory_pivoted['sum'] = df_groupbyCategory_pivoted.sum(axis=1)
    df_groupbyCategory_pivoted = df_groupbyCategory_pivoted.sort_values(['sum'],ascending=False)
    df_groupbyCategory_pivoted.to_sql('ordersgroupbycategory', engine, if_exists='replace')

    df_groupbyUser = df.groupby(['userId','month'])['Amount'].agg(['sum','count'])
    df_groupbyUser.reset_index(inplace=True)
    df_groupbyUser = df_groupbyUser.rename(columns={'sum': 'amount', 'count':'numOrders'})
    df_groupbyUser_numOrders = df_groupbyUser.pivot(index='userId',columns='month',values='numOrders').fillna(value=0)
    df_groupbyUser_amount = df_groupbyUser.pivot(index='userId',columns='month',values='amount').fillna(value=0)
    df_groupbyUser_numOrders.reset_index(inplace=True)
    df_groupbyUser_amount.reset_index(inplace=True)

    df_groupbyUser_numOrders.columns = ['userId']+['{:02d}'.format(int(x))+'-numOrders' for x in df_groupbyUser_numOrders.columns[1:]]
    df_groupbyUser_amount.columns = ['userId']+['{:02d}'.format(int(x))+'-amount' for x in df_groupbyUser_amount.columns[1:]]
    df_groupbyUser_numOrders.to_sql('ordersgroupbyusersnum', engine, if_exists='replace')
    df_groupbyUser_amount.to_sql('ordersgroupbyusersamount', engine, if_exists='replace')


def writeCollectionGroupbyModule():
    engine,conn = connDB()
    """
    Read collections table and write new table collectiongroupby
    :return: format as "userId, moduleName, year, month, count"
    """
    # query:
    sql_query = """
    SELECT * FROM collection
    ;
    """
    df = pd.read_sql_query(sql_query, conn)

    df['month'] = df['created_date'].apply(lambda x: (x.year-2017)*12+x.month) #"added month index 2017-01 as 1"
    df = df.drop(['created_date'], axis=1)
    df_series = df.groupby(['module', 'month'])['itemId'].count()
    df = df_series.to_frame()
    df.reset_index(inplace=True)
    df = df.rename(columns={'itemId': 'numCollections'})
    df_pivoted = df.pivot(index='module',columns='month',values='numCollections').fillna(value=0)
    df_pivoted['sum'] = df_pivoted.sum(axis=1)
    df_pivoted = df_pivoted.sort_values(['sum'],ascending=False)
    df_pivoted.to_sql('collectiongroupbymodule', engine, if_exists='replace')


def writeCollectionGroupbyUserAndModule():
    engine, conn = connDB()
    """
    Read collections table and write new table collectiongroupby
    :return: format as "userId, moduleName, year, month, count"
    """
    # query:
    sql_query = """
    SELECT created_date, "userId", "itemId", module  
    FROM collection
    WHERE created_date >= '2018-01-01'::date
    ;
    """
    df = pd.read_sql_query(sql_query, conn)

    df['month'] = df['created_date'].apply(lambda x: '{:02.0f}'.format((x.year-2017)*12+x.month)+'-collection') #"added month index 2017-01 as 1"
    # df['monthModule']=df[['month', 'module']].apply(lambda x: ''.join(x), axis=1)
    df = df.drop(['created_date'], axis=1)

    df_series = df.groupby(['userId', 'month'])['itemId'].count()
    df = df_series.to_frame()
    df.reset_index(inplace=True)
    df = df.rename(columns={'itemId': 'numCollections'})
    df_pivoted = df.pivot(index='userId',columns='month',values='numCollections').fillna(value=0)
    df_pivoted['sum'] = df_pivoted.sum(axis=1)

    df_pivoted = df_pivoted.sort_values(['sum'],ascending=False)
    df_pivoted = df_pivoted[df_pivoted['sum']>10]
    df_pivoted = df_pivoted.drop('sum', axis=1)
    df_pivoted.to_sql('collectiongroupbyuserandmodule', engine, if_exists='replace')
    print("Wrote to collectiongroupbyuserandmodule")


def writeFeatures():
    """
    Write features. The users are active users.
    Feature time dated to last 12 months.
    :return:
    """
    engine, conn = connDB()
    feature_query = """
        SELECT collectiongroupbyuserandmodule.*, sellers."CreatedDate" AS sellerCreatedDate, 
        ordersgroupbyusersamount.*, ordersgroupbyusersnum.*, wishlistsgroupbyusersnum.*
        FROM collectiongroupbyuserandmodule
        INNER JOIN users ON collectiongroupbyuserandmodule."userId"=users."userId"
        LEFT JOIN sellers ON users."email"=sellers."Email"
        LEFT JOIN ordersgroupbyusersamount on collectiongroupbyuserandmodule."userId"=ordersgroupbyusersamount."userId"
        LEFT JOIN ordersgroupbyusersnum on collectiongroupbyuserandmodule."userId"=ordersgroupbyusersnum."userId"
        LEFT JOIN wishlistsgroupbyusersnum on collectiongroupbyuserandmodule."userId" = wishlistsgroupbyusersnum."userId"      
        ;
    """
    df_features = pd.read_sql_query(feature_query, conn)
    df_features['month'] = df_features['sellercreateddate'].apply(lambda x: (x.year-2017)*12+x.month).fillna(value=0)
    df_features['month'] = df_features['month'].apply(lambda x: int(x))
    df_features = df_features.drop(['sellercreateddate'],axis=1)
    df_features = df_features.fillna(value=0)
    endMonth = int(df_features.columns[-2][0:2])

    featureColumns = ['userId','t-3-collection','t-2-collection','t-1-collection']+['t-3-numOrders','t-2-numOrders','t-1-numOrders']+ \
                     ['t-3-amount', 't-2-amount', 't-1-amount']+['t-3-wishlist', 't-2-wishlist', 't-1-wishlist']
    featureColumns.append('selling')

    # Write the training and test matrix
    # Format: userID, 3-collections, 3-orders, 3-order-amount, 3-wishlist, and isSeller?
    # For a seller who started selling in month 20, will consider all months before 20 (including 20).
    # For a collector who hasn't sold anything. Write features for every month
    features = pd.DataFrame(columns=featureColumns)
    for rowidx in range(df_features.shape[0]):
        sellingMonth = int(df_features.iloc[rowidx, -1])
        t = sellingMonth
        invalid_selling_month = 15 # Consider only sellers started after 2018/03

        # If a collector is a valid seller, the label should be 1 for the starting to sell month
        if sellingMonth>invalid_selling_month:
            # for idx in range(min(3,t-15)):
            end = t
            # Create order columns for each collector
            orderColumn = ['userId']+['{:02d}'.format(x) + '-collection' for x in range(end-3,end)]+\
                          ['{:02d}'.format(x) + '-numOrders' for x in range(end-3,end)]+\
                            ['{:02d}'.format(x) + '-amount' for x in range(end-3,end)]+ \
                          ['{:02d}'.format(x) + '-numWishlist' for x in range(end-3,end)]
            newRowValue = [list(df_features.iloc[rowidx][orderColumn])[0]]
            newRowValue.extend(list(df_features.iloc[rowidx][orderColumn])[4:])
            newRowValue.append(1)
            newRowDF = pd.DataFrame([newRowValue], columns=featureColumns)
            features = features.append(newRowDF,ignore_index=True)

        # If the collector is not a seller, labels are 0
        else:
            t = endMonth
            while t>15:
                orderColumn = ['userId'] + ['{:02d}'.format(x) + '-collection' for x in range(t-3,t)]+\
                              ['{:02d}'.format(x) + '-numOrders' for x in range(t - 3, t)] + \
                              ['{:02d}'.format(x) + '-amount' for x in range(t - 3, t)]+\
                                ['{:02d}'.format(x) + '-numWishlist' for x in range(t - 3, t)]
                newRowValue = [list(df_features.iloc[rowidx,:][orderColumn])[0]]
                newRowValue.extend(list(df_features.iloc[rowidx][orderColumn])[4:])
                newRowValue.append(0)
                newRowDF = pd.DataFrame([newRowValue], columns=featureColumns)
                features = features.append(newRowDF,ignore_index=True)
                t -= 1

    features.to_sql('features', engine, if_exists='replace')

    # Write a new table featuresrecent3month for prediction
    features_recent_3mon = pd.DataFrame(columns=featureColumns[:-1])
    for rowidx in range(df_features.shape[0]):
        #pass the sellers
        if int(df_features.iloc[rowidx, -1]):
            continue
        orderColumn = ['userId']+['{:02d}'.format(x) + '-collection' for x in range(endMonth-3,endMonth)]+\
                      ['{:02d}'.format(x) + '-numOrders' for x in range(endMonth-3,endMonth)]+\
                        ['{:02d}'.format(x) + '-amount' for x in range(endMonth-3,endMonth)]+ \
                      ['{:02d}'.format(x) + '-numWishlist' for x in range(endMonth-3,endMonth)]
        newRowValue = [list(df_features.iloc[rowidx][orderColumn])[0]]
        newRowValue.extend(list(df_features.iloc[rowidx][orderColumn])[4:])
        newRowDF = pd.DataFrame([newRowValue], columns=featureColumns[:-1])
        features_recent_3mon = features_recent_3mon.append(newRowDF,ignore_index=True)
    features_recent_3mon.to_sql('featuresrecent3month', engine, if_exists='replace')


def writeSummary():
    """
    Aggregate data from three tables: collectiongroupbymodule, orders, and wishlist
    The output table:
                    09  10  11 ...
    neworders
    newcollections
    newwishlist
    """
    engine,conn = connDB()
    # write collection summary
    collections_number_query = """
    select * from collectiongroupbymodule;
    """
    df_collections = pd.read_sql_query(collections_number_query, conn)
    df_collections = df_collections.sum(axis=0)
    df_collections = [int(x/1000) for x in list(df_collections)[-13:-1]]

    # write order summary
    order_number_query = """
    select * from orders;
    """
    df_orders = pd.read_sql_query(order_number_query, conn)
    df_orders['month'] = df_orders['created_date'].apply(lambda x: (x.year-2017)*12+x.month)
    df_orders = list(df_orders.groupby(['month'])['userId'].count())[-12:]
    df_orders = [x for x in list(df_orders)[-13:-1]]

    wishlist_query="""
    select * from wishlistgroupbycategory;
    """
    df_wishlist = pd.read_sql_query(wishlist_query, conn)
    df_wishlist = df_wishlist.sum(axis=0)
    df_wishlist = [int(x/1000) for x in list(df_wishlist)[-13:-1]]

    #write user summary
    user_number_query = """
    SELECT * FROM users;
    """
    df_user = pd.read_sql_query(user_number_query, conn)
    user_groupby=df_user.groupby(['month'])['userId'].count()
    df_user = [int(x/1000) for x in list(user_groupby)[-12:]]

    # write seller summary
    seller_number_query = """
    SELECT * FROM sellers;
    """
    df_seller = pd.read_sql_query(seller_number_query, conn)
    df_seller['month'] = df_seller['CreatedDate'].apply(lambda x: (x.year-2017)*12+x.month)
    df_seller = df_seller.groupby(['month'])['userId'].count()
    monthList = list(df_seller.index)[-12:]
    df_seller = list(df_seller)[-12:]

    df_summary = pd.DataFrame.from_dict({'NumberOfOrders':df_orders, 'NumberOfCollections':df_collections, \
                                         'NumberOfWishlist':df_wishlist, 'NumberOfUsers':df_user, \
                                         'NumberOfSellers':df_seller, }, orient='index',columns=monthList)
    df_summary.to_sql('summary', engine, if_exists='replace')


def main():
    """
    Cache module write SQL tables for updating dashboard and model training.
    Cache doesn't return anything; it only writes to SQL
    dataIngestion module read the cache SQL and return DF for plot in Dashboard.
    """
    writeOrderGroupby()
    writeCollectionGroupbyUserAndModule()
    writeCollectionByUser()
    writeSummary()
    writeInventory()
    write_wanttoinv()
    writeWishlistGroupby()
    writeCollectionGroupbyUserAndModule()
    writeFeatures()
    writeCollectionGroupbyModule()


if __name__ == "__main__":
    main()
