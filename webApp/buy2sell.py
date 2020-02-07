""" Main module supporting the webapp (tianshi-wang.com)

The data warehouse is hosted for this webapp

"""

import os
import copy
import datetime as dt

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from webApp import dashInterface
from webApp.controls import CATEGORY_NAME, CATEGORY_COLORS


#
# server = Flask(__name__)
# @server.route('/')
# def index():
#     return "hellow world"
#
#
# # Initialize the dash server and set up the style format
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(server=server, url_base_pathname='/dash/', external_stylesheets=external_stylesheets)
# app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})
#
#
# # app.config.suppress_callback_exceptions = True
# app.title = "Tianshi Wang's website"
# CORS(server)
#
# if 'DYNO' in os.environ:
#     app.scripts.append_script({
#         'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'  # noqa: E501
#     })
#


# # Create controls
category_name_options = [{'label': str(CATEGORY_NAME[category_name]),
                        'value': str(category_name)}
                       for category_name in CATEGORY_NAME]


# Create global chart template
layout = dict(
    autosize=True,
    height=500,
    font=dict(color='#CCCCCC',size='12'),
    titlefont=dict(color='#CCCCCC', size='18'),
    margin=dict(
        l=45,
        r=35,
        b=50,
        t=45
    ),
    hovermode="closest",
    plot_bgcolor="#191A1A",
    paper_bgcolor="#020202",
    legend=dict(font=dict(size=12), orientation='h'),
    yaxis=dict(title='')
)


def create_buy2sell(app):
    app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})

    # app.config.suppress_callback_exceptions = True
    app.title = "Tianshi Wang's website: Buy2Sell project at Insight"
    if 'DYNO' in os.environ:
        app.scripts.append_script({
            'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
        # noqa: E501
        })
    app.layout = get_layout()
    register_callbacks(app)



def get_header():
    """
    Return the header including components:
    1. Description of the webapp
    2. Links to Github, Linkedin and personal website
    """
    header=html.Div(
        [
            html.Div(
                [
                    html.H1(
                        'Buy2Sell',
                        style={
                            'textAlign': 'center',
                            'color': 'white',
                            'margin-top': '20',
                        }
                    ),
                    html.H3(
                        'Converting Collectors to Sellers in an Online Collectible Marketplace',
                        style={
                            'textAlign': 'center',
                            'color': 'white'
                        }
                    ),
                    html.P(
                        'Tianshi Wang\'s Project as a Data Science Fellow at Insight Data Science',
                        style={
                            'textAlign': 'center',
                            'color': 'white',
                        }
                    ),
                    html.Div(
                        [
                            html.P("   ", className='three columns',style={'textAlign': 'center','color': 'white'}),
                            html.A("LinkedIn", href='https://www.linkedin.com/in/tianshi-wang/', target="_blank",
                                   className='two columns',style={'textAlign': 'center','color': 'white'}),
                            html.A("GitHub", href='https://github.com/tianshi-wang/Buy2Sell_Insight', target="_blank",
                                   className='two columns',style={'textAlign': 'center','color': 'white'}),
                            html.A("Personal Website", href='https://tswang.wixsite.com/home', target="_blank",
                                   className='two columns',style={'textAlign': 'center','color': 'white'}),
                            html.P("   ", className='three columns', style={'textAlign': 'center', 'color': 'white'}),
                        ],
                        className='row',
                        style={'marginBottom': 40}
                    )
                ],
                className='twelve columns',
                style={'backgroundColor':'#1D1E1E'}
            ),

        ],
    )
    return header


# The layout of the whole webapp

def get_layout():
    return html.Div(
        [   #Top text and logo
            get_header(),

            # Description of Figure 1 (by-category inventory level) and Fig.2 (historic inventory)
            html.Div(
                [
                    html.H2(
                        'By-category Inventory Level',
                        className='eight columns',
                        style={'backgroundColor': '#F7F7F8',
                               'margin-top': '40',
                               'margin-left':'20'}
                    ),
                    html.Div(
                        [
                            html.P(
                                'Below diagrams shows the inventory level, defined by the number of new wish list over inventory, \
                                for each category.',
                            ),
                            html.P(
                                ' The right image show the inventory level history for the selected category on the left.',
                            ),
                        ],
                        className='eight columns',
                        style={'margin-left': '20'}
                    ),
                    html.Img(
                        src="https://www.covetly.com/Content/images/covetly-logo-trans-with-slight-space-top-and-bottom.png",
                        className='two columns',
                        style={
                            'height': '80',
                            'width': '220',
                            'float': 'right',
                            'position': 'relative',
                            'overflow': 'auto',
                            'margin-right':'20',
                        },
                    ),
                ],
                className='row',
                style={'backgroundColor': '#F7F7F8',
                       'margin-top': '40',
                       }
            ),

            # Plot Fig.1 (category_inventory_graph) and Fig. 2 (historic_inventory_graph)
            # The cursor position on Fig.2 determines the category to plot on Fig.2
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(id="category_inventory_graph")
                        ],
                        className='eight columns',
                        style={'margin-top': '20', 'marginBottom': 40}
                    ),
                    html.Div(
                        [
                            dcc.Graph(id='historic_inventory_graph')
                        ],
                        className='four columns',
                        style={'margin-top': '20', 'marginBottom': 40}
                    ),
                ],
            ),

            # Description of the table for suggested collectors
            html.Div(
                [
                    html.H2(
                        'Suggested Sellers',
                        className='twelve columns',
                        style={'margin-left':'20'},
                    ),
                    html.P(
                        'Suggest a list of potential sellers. The score is defined by: ',
                        className='twelve columns',
                        style={'margin-left': '20'},
                    ),
                    html.P(
                        'score = likelihood* Σ(category_inventory_percentage),',
                        className='twelve columns',
                        style={'margin-left': '20', 'textAlign': 'center'},
                    ),
                    html.P(
                        'where the likelihood is the possibility that the user would sell next month and the category_inventory_percentage\
                        stands for the percentage of the selected categories in the user\'s whole collection.',
                        className='twelve columns',
                        style={'margin-left': '20'},
                    ),
                    html.P(
                        'By default, the selected categories are the three with lowest inventory level. The filter can be used\
                         for choosing categories.',
                        className='twelve columns',
                        style={'margin-left': '20'},
                    ),
                ],
                className='row',
                style = {'backgroundColor': '#F7F7F8','margin-top': '40'}
            ),

            # Description of the drop-off (category-selector)
            html.Div(
                [
                    html.P(
                        'Filter by categories:',
                        # className='two columns'
                        className='eight columns',
                        style={'margin-left': '20'}
                    ),
                ],
                className='twelve columns',
                style={'backgroundColor': '#F7F7F8', },
            ),

            # Layout of the category selector (RadioItems and drop-down)
            # By default, the radio items shows 'Low-inventory categories (top3)'
            # which is the three categories with lowest inventory levels on Fig.1 (category_inventory_level)
            # Users can also add or delete the selected categories using drop-sown
            # The change of the category selector will update the table
            html.Div(
                [

                    dcc.RadioItems(
                        id='category_name_selector',
                        options=[
                            {'label': 'All   ', 'value': 'all'},
                            {'label': 'Low-inventory categories (top3)', 'value': 'LowInventory(top3)'},
                        ],
                        value='LowInventory(top3)',
                        labelStyle={'display': 'inline'},
                        style={'textAlign': 'left',
                               'marginBottom': 10, 'margin-left': '20'},
                    ),

                    dcc.Dropdown(
                        id='category_name_dropdown',
                        options=category_name_options,
                        multi=True,
                        value=[],
                    ),
                ],
                className='twelve columns',
                style={'backgroundColor': '#F7F7F8'}
            ),
            html.Div(
                [
                    html.H2(
                    )
                ],
                className='row',
                style={'backgroundColor': '#F7F7F8',
                       'margin-top': '40',
                       }
            ),

            html.Div(
                [
                    dcc.Graph(id="table-graph")
                ],
                     className="row",
                     # style={"padding": 20}
                     ),

            # Description for the business overview section
            html.Div(
                [
                    html.H2(''),
                    html.H2(
                        'Business Overview ',
                        className='twelve columns',
                        style={'text-align': 'center','margin-top': '40', 'marginBottom': 20},
                    ),
                ],
                className='row'
            ),

            # Plot Fig.3 (summary_graph) showing the number of new collections, wishlist and inventory
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(id="summary_graph")
                        ],
                        className='twelve columns',
                        style={'margin-top': '20', 'marginBottom': 40}
                    ),
                ],
            ),

            # Description of the Users and Sellers section
            html.Div(
                [
                    html.H1(''),
                    html.H1(
                        'Users and Sellers',
                        className='eight columns',
                    ),
                ],
                className='row'
            ),

            # Plot of the number of new Users and Sellers
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(id='user_seller_graph')
                        ],
                        className='twelve columns',
                        style={'margin-top': '20', 'marginBottom': 40}
                    ),
                ]
            ),
        ],
        className='ten columns offset-by-one',
    )




def register_callbacks(app):
    # Implementation of each components in layout
    # The order of the functions is the same as webapp display
    @app.callback(Output('category_inventory_graph', 'figure'),
                  [Input('category_name_dropdown', 'value')]
                  )
    def make_category_inventory_figure(category_name_dropdown):
        """
        Plot the by-category inventory level graph
        :param category_name_dropdown: Selected categores (not necessary)
        :return: by-category inventory level graph
        """
        data=[]
        category_inventory_df= dashInterface.inventoryLevel()
        print(category_inventory_df)
        data.append(dict(
            type='bar',
            name=category_name_dropdown[0],
            x = category_inventory_df.iloc[:,1],
            y = category_inventory_df.iloc[:,-1]*100,
            line=dict(
                shape="spline",
                smoothing=2,
                width=1,
                color=CATEGORY_COLORS[0]
            ),
            bargap=0.4,
            opacity=0.8,
            marker=dict(symbol='diamond-open', color='#59C3C3')
            ))

        layout_individual = copy.deepcopy(layout)
        print(layout_individual)
        layout_individual['title'] = 'By-category Inventory Level'  # noqa: E501
        layout_individual['yaxis']['title'] = 'inventory/wishlist (%)'
        figure = dict(data=data, layout=layout_individual)
        return figure

    @app.callback(Output('historic_inventory_graph', 'figure'),
                  [Input('category_inventory_graph', 'hoverData'),])
    def make_historic_inventory_graph(category_inventory_graph_hover):
        """
        Plot historic_inventory_graph
        :param category_inventory_graph_hover: The cursor position on Fig.1 (by-category inventory)
        :return: historic_inventory_graph
        """
        if category_inventory_graph_hover is None:
            category_inventory_graph_hover = {'points': [{'x': 'funko'}]}
        print(category_inventory_graph_hover)
        chosen = [point['x'] for point in category_inventory_graph_hover['points']]
        chosenName = chosen[0]
        category_inventory_df= dashInterface.inventoryLevel()
        category_inventory_df = category_inventory_df[category_inventory_df["module"]==chosenName].iloc[:,1:]

        # Plot collection to data
        print(category_inventory_df)
        colors = ['#F9ADA0', '#849E68', '#59C3C3','#67BD65','#FDBF6F',]

        # Append plot-dictionary to data
        data=[]
        data.append(dict(
            type='scatter',
            mode='lines+markers',
            name=chosenName,
            x=[dt.datetime(year=2017+int(int(x)/12),month=1+int(int(x)%12),day=1) for x in category_inventory_df.columns[1:]],
            y=[y*100 for y in category_inventory_df.iloc[0,1:]],
            line=dict(
                shape="spline",
                smoothing=2,
                width=2,
                color=colors[0]
            ),
            marker=dict(symbol='diamond-open')
        ))

        layout_individual = copy.deepcopy(layout)
        layout_individual['title'] = '%s Inventory History' %(chosenName.capitalize())  # noqa: E501
        layout_individual['yaxis']['title'] = 'inventory/wishlist (%)'
        figure = dict(data=data, layout=layout_individual)
        return figure


    @app.callback(Output('category_name_dropdown', 'value'),
                  [Input('category_name_selector', 'value')])
    def display_type(selector):
        """
        Display the radio-item selector
        If selector == 'LowInventory(top3)', select three categories with lowest inventory levels
        Check dashInterface.top3lowest for details
        """
        if selector == 'all':
            return list(CATEGORY_NAME.keys())
        elif selector == 'LowInventory(top3)':
            return dashInterface.top3lowest()


    @app.callback(Output("table-graph", "figure"),
                  [Input('category_name_dropdown', 'value')],)
    def update_table(category_name_dropdown):
        """
        For user selections, return the relevant table
        Input: the category_selector (radioitem + dropdown)
        """
        selectedCategories = [CATEGORY_NAME[idx] for idx in category_name_dropdown]
        # Query collector score for selected categories
        # Check dashInterface.userTable for details
        df = dashInterface.userTable(tuple(selectedCategories))
        value_header = []
        value_cell = []
        for col in df.columns:
            value_header.append(col)
            value_cell.append(df[col])
        trace = go.Table(
            header={"values": value_header, "fill": {"color": "grey"}, "align": ['center'], "height": 30,
                    "line": {"width": 2, "color": "#685000"}, "font": {"size": 15, "color":'white'}},
            cells={"values": value_cell, "fill": {"color": "#F7F7F8"}, "align": ['left', 'center'],
                   "line": {"color": "#685000"}})
        layout = go.Layout(title=f"List of Rcommended Sellers <br> (You may use the dropdown above to select categories)", height=700)
        return {"data": [trace], "layout":layout}


    @app.callback(Output('summary_graph', 'figure'),
                  [Input('category_name_dropdown', 'value')],
                  [State('summary_graph', 'relayoutData')])  # No input this time. [Input('main_graph', 'hoverData')]
    def make_summary_figure(userId,summary_graph_layout):
        """
        Plot the trend for newly added collections, wishlist, and inventories
        """
        data=[]
        summary_df = dashInterface.summary()
        names=['New orders', 'New Collections (k)', 'New Wishlist (k)']
        colors=['#F9ADA0','#849E68','#59C3C3','#fac1b7']
        for idx in range(3):
            data.append(dict(
                type='scatter',
                mode='lines+markers',
                name=names[idx],
                # Data was stored as 1 for 2017-01. The first column is categories. Plot from the second column
                x=[dt.datetime(year=2017+int(int(x)/12),month=1+int(int(x)%12),day=1) for x in summary_df.columns[1:]],
                y=[int(y) for y in summary_df.iloc[idx,1:]],
                line=dict(
                    shape="spline",
                    smoothing=2,
                    width=1,
                    color= colors[idx]
                ),
                marker=dict(symbol='diamond-open')
            ))

        layout_individual = copy.deepcopy(layout)
        layout_individual['title'] = 'Business Overview'  # noqa: E501
        figure = dict(data=data, layout=layout_individual)
        return figure



    @app.callback(Output('user_seller_graph', 'figure'),
                  [Input('category_name_dropdown', 'value')])   # No input this time. [Input('main_graph', 'hoverData')]
    def make_user_seller_graph(summary_graph_hover):
        """
        Plot the trend of new collectors and new sellers
        """
        data=[]
        df_user_seller = dashInterface.summary()
        # The 3:5 are rows for new_users and new_sellers
        df_user_seller = df_user_seller.iloc[3:5,:]
        names=['New users', 'New sellers']
        colors=['#59C3C3','#fac1b7']

        data.append(dict(
            type='scatter',
            mode='lines+markers',
            name='New users (*100)',
            # The data use 2017-01 as 1
            x=[dt.datetime(year=2017+int(int(x)/12),month=1+int(int(x)%12),day=1) for x in df_user_seller.columns[1:]],
            y=[int(y*10) for y in df_user_seller.iloc[0,1:]],
            line=dict(
                shape="spline",
                smoothing=2,
                width=1,
                color= colors[0]
            ),
            marker=dict(symbol='diamond-open')
        ))

        data.append(dict(
            type='scatter',
            mode='lines+markers',
            name='New sellers',
            x=[dt.datetime(year=2017+int(int(x)/12),month=1+int(int(x)%12),day=1) for x in df_user_seller.columns[1:]],
            y=[int(y) for y in df_user_seller.iloc[1,1:]],
            line=dict(
                shape="spline",
                smoothing=2,
                width=1,
                color= colors[1]
            ),
            marker=dict(symbol='diamond-open')
        ))

        layout_individual = copy.deepcopy(layout)
        layout_individual['title'] = 'New users and sellers'  # noqa: E501
        figure = dict(data=data, layout=layout_individual)
        return figure


# The following functions not implemented in the current webapp
# @app.callback(Output('byCategory_graph', 'figure'),
#               [Input('summary_graph', 'hoverData')])   # No input this time. [Input('main_graph', 'hoverData')]
# def make_byCategory_graph(summary_graph_hover):
#     if summary_graph_hover is None:
#         summary_graph_hover = {'points': [{'curveNumber': 0,
#                                         'pointNumber': 569,
#                                         'customdata': 31101173130000}]}
#     chosenFigure = 1   # Initialize figure to show as new orders by category
#     chosen = [point['curveNumber'] for point in summary_graph_hover['points']]
#     chosenFigure = chosen[0]
#
#     # Plot collection to data
#     colors = ['#F9ADA0', '#849E68', '#59C3C3','#67BD65','#FDBF6F',]
#
#     # Plot user to data1
#     data0=[]
#     users_df = dashInterface.ordersGroupbyCategory()
#     names = list(users_df['CategoryName'])
#     for idx in range(users_df.shape[0]):
#         data0.append(dict(
#             type='scatter',
#             mode='lines+markers',
#             name=names[idx],
#             x=[dt.datetime(year=2017+int(int(x)/12),month=1+int(int(x)%12),day=1) for x in users_df.columns[1:]],
#             y=[int(y) for y in users_df.iloc[idx,1:]],
#             line=dict(
#                 shape="spline",
#                 smoothing=2,
#                 width=1,
#                 color=colors[idx]
#             ),
#             marker=dict(symbol='diamond-open')
#         ))
#
#
#     data1=[]
#     collections_df = dashInterface.collectionGroupbyModule()
#     names = list(collections_df.module)
#     for idx in range(collections_df.shape[0]):
#         data1.append(dict(
#             type='scatter',
#             mode='lines+markers',
#             name=names[idx],
#             x=[dt.datetime(year=2017+int(int(x)/12),month=1+int(int(x)%12),day=1) for x in collections_df.columns[1:-1]],
#             y=[int(y) for y in collections_df.iloc[idx,1:-1]],
#             line=dict(
#                 shape="spline",
#                 smoothing=2,
#                 width=1,
#                 color=colors[idx]
#             ),
#             marker=dict(symbol='diamond-open')
#         ))
#
#     data2=[]
#     users_df = dashInterface.wishlistGroupbyModule()
#     names = list(users_df['CategoryName'])
#     for idx in range(users_df.shape[0]):
#         data2.append(dict(
#             type='scatter',
#             mode='lines+markers',
#             name=names[idx],
#             x=[dt.datetime(year=2017+int(int(x)/12),month=1+int(int(x)%12),day=1) for x in users_df.columns[1:]],
#             y=[int(y) for y in users_df.iloc[idx,1:]],
#             line=dict(
#                 shape="spline",
#                 smoothing=2,
#                 width=1,
#                 color=colors[idx]
#             ),
#             marker=dict(symbol='circle-open')
#         ))
#
#     data=[data0, data1, data2]
#
#     layout_individual = copy.deepcopy(layout)
#     layout_individual['title'] = ['New Collections by Category','New Orders by Category', 'New Wishlist by Category'][chosenFigure]  # noqa: E501
#     figure = dict(data=data[chosenFigure], layout=layout_individual)
#     return figure