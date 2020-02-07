import os
import copy
import datetime as dt
import dash


from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from webApp import dashInterface


# Create global chart template
# layout = dict(
#     autosize=True,
#     height=500,
#     font=dict(color='#CCCCCC',size='12'),
#     titlefont=dict(color='#CCCCCC', size='18'),
#     margin=dict(
#         l=45,
#         r=35,
#         b=50,
#         t=45
#     ),
#     vertical_align="middle",
#     hovermode="closest",
#     plot_bgcolor="#191A1A",
#     paper_bgcolor="#020202",
#     legend=dict(font=dict(size=12), orientation='h'),
#     yaxis=dict(title='')
# )


def create_index(app):
    app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})

    # app.config.suppress_callback_exceptions = True
    app.title = "Tianshi Wang's website"
    if 'DYNO' in os.environ:
        app.scripts.append_script({
            'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
        })
    app.layout = get_layout()

def get_header():
    """
    Return the header including components:
    1. Description of the webapp
    2. Links to Github, Linkedin and personal website
    """
    header=html.Div(
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
        className='row',
        style={'backgroundColor':'#1D1E1E'}
    )
    return header



def get_layout():
    return html.Div(
        [   #Top text and logo
            get_header(),

            # Description of Figure 1 (by-category inventory level) and Fig.2 (historic inventory)
            html.Div(
                [
                    html.Div(
                        [
                            html.H2(
                                'By-category Inventory Level',

                            )
                        ],
                        className='eight columns',
                        style={
                            # 'backgroundColor': '#F7F7F8',
                            #    'margin-top': '40',
                               }
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
                        style={}
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
                            'margin-right':'auto',
                        },
                    ),
                ],
                className='twelve columns',
                style={'backgroundColor': '#F7F7F8',
                       'margin-top': '40',
                       }
            )
        ],
        className='row',
        style={
            'width': '80%',
            'marginLeft': 40, 'marginRight': 40, 'marginTop': 20, 'marginBottom': 20,
            'display': 'inline-block',
        }
    )