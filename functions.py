import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


@st.cache
def load_data(filename):
    df = pd.read_csv(filename, index_col=0)
    return df

def gaugechart(key_data):
    val = key_data['proba']
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value= val,
        domain={'x': [0, 1], 'y': [0,1]},
        title={'text': "Probability to obtain Credit"}))
    fig.update_traces(number_valueformat = ".1%",gauge_axis_tickmode='array',gauge_axis_range=[0,1])
    fig.update_layout(autosize=False,width=600,height=400)
    if val < 0.4:
        fig.update_traces(gauge_bar_color = 'red')
    else:
        if val > 0.7:
            fig.update_traces(gauge_bar_color = 'green')
        else: fig.update_traces(gauge_bar_color = 'orange')
    return fig

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

def comparisonchart(df, key_data):

    # Data prep
    ed_df = df[df['NAME_EDUCATION_TYPE'] == key_data['education_type']].copy()
    gender_df = df[df['CODE_GENDER'] == key_data['gender']].copy()
    age_df = df[df['age_group'] == key_data['age_group']].copy()
    debt_df = df[df['debt_group'] == key_data['debt_group']].copy()

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[2,2],
        subplot_titles=("Within the same Age Group","With the same Education Level", "With a similar Debt/Income ratio", "With the same Gender"),
        specs=[[{'type' : 'domain'}, {'type' : 'domain'}],
               [{'type' : 'domain'},{'type' : 'domain'}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    fig.add_trace(go.Indicator(
        value = age_df['proba'].mean(),
        delta = {'reference': 2*age_df['proba'].mean()-key_data['proba'], 'valueformat': '.2%'}),
        row = 1,
        col = 1
    )

    if age_df['proba'].mean() < 0.4:
        fig.update_traces(gauge_bar_color = 'red')
    else:
        if age_df['proba'].mean() > 0.7:
            fig.update_traces(gauge_bar_color = 'green')
        else: fig.update_traces(gauge_bar_color = 'orange')
    fig.update_traces(number_valueformat=".1%", gauge_axis_tickmode='array', gauge_axis_range=[0, 1])

    fig.add_trace(go.Indicator(
        value=ed_df['proba'].mean(),
        delta={'reference': 2*ed_df['proba'].mean()-key_data['proba'], 'valueformat':'.2%'}),
        row=1,
        col=2
    )

    if ed_df['proba'].mean() < 0.4:
        fig.update_traces(gauge_bar_color='red')
    else:
        if ed_df['proba'].mean() > 0.7:
            fig.update_traces(gauge_bar_color='green')
        else:
            fig.update_traces(gauge_bar_color='orange')

    fig.update_traces(number_valueformat=".1%", gauge_axis_tickmode='array', gauge_axis_range=[0, 1])

    fig.add_trace(go.Indicator(
        value=debt_df['proba'].mean(),
        delta={'reference': 2*debt_df['proba'].mean()-key_data['proba'], 'valueformat':'.2%'}),
        row=2,
        col=1
    )

    if debt_df['proba'].mean() < 0.4:
        fig.update_traces(gauge_bar_color='red')
    else:
        if debt_df['proba'].mean() > 0.7:
            fig.update_traces(gauge_bar_color='green')
        else:
            fig.update_traces(gauge_bar_color='orange')

    fig.update_traces(number_valueformat=".1%", gauge_axis_tickmode='array', gauge_axis_range=[0, 1])

    fig.add_trace(go.Indicator(
        value=gender_df['proba'].mean(),
        delta={'reference': 2*gender_df['proba'].mean()-key_data['proba'], 'valueformat':'.2%'}),
        row=2,
        col=2
    )

    if gender_df['proba'].mean() < 0.4:
        fig.update_traces(gauge_bar_color='red')
    else:
        if gender_df['proba'].mean() > 0.7:
            fig.update_traces(gauge_bar_color='green')
        else:
            fig.update_traces(gauge_bar_color='orange')

    fig.update_traces(number_valueformat=".1%", gauge_axis_tickmode='array', gauge_axis_range=[0, 1])

    fig.update_layout(
        template={'data': {'indicator': [{'mode': "number+delta+gauge"}]}},
        autosize=False,
        width=1000,
        height=800
    )

    return fig