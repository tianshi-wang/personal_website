"""This module supports the webapp main module-app.py
It contains internal dictionaries and lists.
"""

# List of colors for plot
CATEGORY_COLORS = ['#FFEDA0', '#FA9FB5','#A1D99B','#67BD65','#BFD3E6','#B3DE69','#FDBF6F','#FC9272',\
    '#D0D1E6','#ABD9E9','#3690C0','#F87A72','#CA6BCC','#DD3497','#4EB3D3','#FFFF33','#FB9A99','#A6D853','#D4B9DA',\
    '#AEB0B8','#CCCCCC','#EAE5D9','#C29A84','#FA9FB5','#A1D99B','#67BD65','#BFD3E6','#B3DE69','#FDBF6F','#FC9272']


# In next version, the key remains to be the SEO-category-name
# Value will be display name on the website
# In this version, the display_name has not been given yet.
CATEGORY_NAME = {"funko":"funko",
    "mystery-minis":"mystery-minis",
    "dorbz":"dorbz",
    "amiibo":"amiibo",
    "rock-candy":"rock-candy",
    "marvel-legends":"marvel-legends",
    "vynl":"vynl",
    "disney-infinity-figures":"disney-infinity-figures",
    "tokidoki":"tokidoki",
    "hikari":"hikari",
    "funko-other":"funko-other",
    "kid-robot":"kid-robot",
    "pint-sized-heroes":"pint-sized-heroes",
    "my-little-pony":"my-little-pony",
    "kaws":"kaws",
    "mighty-jaxx":"mighty-jaxx",
    "skylanders":"skylanders",
    "berbrick":"berbrick",
    "kid-robot-blind-boxes":"kid-robot-blind-boxes",
    "masters-of-the-universe":"masters-of-the-universe",
    "star-wars-kenner":"star-wars-kenner",
    "amiibo-cards":"amiibo-cards",
    "pokemon-cards":"pokemon-cards",
    "superplastic":"superplastic",
    "marvel-comics":"marvel-comics",
    "dc-comics":"dc-comics",
    "gi-joe":"gi-joe",
    "covetly-store":"covetly-store"}