
from pymongo import MongoClient


def conn(pwInv, pwDomain):
    """
    Connect and download tables from Client's MongoDB to local DB

    Tables on domain DB: ['collectionitems', 'featuredassets', 'featureditems', 'objectlabs-system',
    'wantlistitems', 'livebids', 'sellers', 'buybids', 'storeadminusers', 'orders', 'inventoryitems, , 'users']

    Tables on categoryDB: ['modules', 'moduleitems', 'objectlabs-system.admin.collections', 'system.indexes',
    'objectlabs-system', 'modulecategories']
    """
    category_client = MongoClient('mongodb://%s@ds054069-a1.zwn31.fleet.mlab.com:54062/covetly-domain-inventory-main?ssl=true' %(pwInv))
    domain_client=MongoClient('mongodb://%s@ds034887-a1.zwn31.fleet.mlab.com:34882/covetly-domain?ssl=true' %(pwDomain))

    domainDB=domain_client['covetly-domain']
    categoryDB=category_client['covetly-domain-inventory-main']
    return categoryDB, domainDB




