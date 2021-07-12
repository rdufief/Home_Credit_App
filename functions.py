import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def load_data(filename):
    df = pd.read_csv(filename, index_col=0)
    return df


def barcharts(df, key_data):
    # MAKE SUBPLOTS
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[2,2],
        specs=[[{"type": "bar"},{"type": "bar"}]],
        subplot_titles=("Against all clients","Against similar clients"),
        vertical_spacing=0.1, horizontal_spacing=0.09)
     # STACKED BAR
    # Get probability distribution
    proba_groups = pd.DataFrame(df.proba_group.value_counts())
    proba_groups = proba_groups.sort_index()

    # Get probability distribution within the client's cluster
    proba_groups2 = pd.DataFrame(df[df['cluster'] == key_data['cluster']].proba_group.value_counts())
    proba_groups2 = proba_groups2.sort_index()

    #Create a list with only the client proba bins_values for charting
    client_proba = ['','','','','','','','','','']
    client_proba2 = client_proba.copy()
    position = proba_groups.index.tolist().index(key_data['proba_bin'])
    client_proba[position] = proba_groups.loc[key_data['proba_bin'],'proba_group']
    client_proba2[position] = proba_groups2.loc[key_data['proba_bin'], 'proba_group']

    proba = key_data['proba']

    # plot params
    labels = proba_groups.columns

    fig.update_layout(autosize=False,
                      width=1200,
                      height=400)
    for i, label_name in enumerate(labels):
        x = df.iloc[:, i].index
        # bar chart to represent clients distribution by probability bin
        fig.add_trace(go.Bar(x=proba_groups.index,
                             y=proba_groups.iloc[:, i],
                             name='Credit Score Bin',
                             hovertemplate='<b>Proba: %{x}%</b><br>#Nb of clients: %{y:,.0f}',
                             legendgroup='grp2',
                             showlegend=True),
                             row=1, col=1)
        # bar chart to represent client position in the distribution
        fig.add_trace(go.Scatter(x=proba_groups.index,
                                 y=client_proba,
                                 mode='markers',
                                 marker_size = 20,
                                 marker_symbol='star-dot',
                                 name='Credit Score',
                                 hovertemplate=f"<b>Client's probability: {proba*100:.2f}%</b>",
                                 showlegend=False),
                                 row=1, col=1)

        # bar chart to represent clients distribution by probability bin, against similar clients (in the same cluster)
        fig.add_trace(go.Bar(x=proba_groups2.index,
                             y=proba_groups2.iloc[:, i],
                             name='Credit Score Bin',
                             hovertemplate='<b>Proba: %{x}%</b><br>#Nb of clients: %{y:,.0f}',
                             legendgroup='grp2',
                             showlegend=True),
                             row = 1, col = 2)

        # bar chart to represent client position in the cluster distribution
        fig.add_trace(go.Scatter(x=proba_groups.index,
                                 y=client_proba2,
                                 mode='markers',
                                 marker_size = 20,
                                 marker_symbol='star-dot',
                                 marker_color='yellow',
                                 name='Credit Score',
                                 hovertemplate=f"<b>Client's probability: {proba*100:.2f}%</b>",
                                 showlegend=False),
                                 row=1, col=2)
    fig.update_yaxes(title_text='Nb of clients', linecolor='grey', mirror=True,
                     title_standoff=0, gridcolor='grey', gridwidth=0.1,
                     zeroline=False,
                     row=1, col=1)
    fig.update_xaxes(title_text='Credit Acceptance Probability',linecolor='grey', mirror=True,
                     row=1, col=1)
    fig.update_xaxes(title_text='Credit Acceptance Probability',linecolor='grey', mirror=True,
                     row=1, col=2)

    return fig