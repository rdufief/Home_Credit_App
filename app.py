
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

from functions import barcharts

#--------------------------
# ICON & TITLE
#--------------------------

st.set_page_config(page_title="Home Credit - Client Scoring",
                   page_icon=":money_with_wings:",
                   layout='wide')


#--------------------------
# VARIABLES & DATA PREP
#--------------------------

path = "/home/romain/Bureau/Formation OpenClassrooms/Projet 07 - Implémentez un modèle de scoring/"

@st.cache
def load_data():
    df = pd.read_csv(path+"customers_data.csv")
    return df


# Will only run once if already cached
df = load_data()

# Add a selectbox to the sidebar:
select_client = st.sidebar.selectbox(
    'Choose client ID',
    (np.unique(df["SK_ID_CURR"]))
)
client_index = df[df["SK_ID_CURR"] == select_client].index[0]

key_data = {
    'client_id': df.loc[client_index,'SK_ID_CURR'],
    'gender': df.loc[client_index,'CODE_GENDER'],
    'age': -df.loc[client_index,'DAYS_BIRTH']/365,
    'nb_children': df.loc[client_index,'CNT_CHILDREN'],
    'family_status': df.loc[client_index,'NAME_FAMILY_STATUS'],
    'education_type': df.loc[client_index,'NAME_EDUCATION_TYPE'],
    'income_tot': df.loc[client_index,'AMT_INCOME_TOTAL'],
    'occupation_type': df.loc[client_index, 'OCCUPATION_TYPE'],
    'income_type': df.loc[client_index, 'NAME_INCOME_TYPE'],
    'owns_car': df.loc[client_index,'FLAG_OWN_CAR'],
    'owns_realty': df.loc[client_index,'FLAG_OWN_REALTY'],
    'credit_amount': df.loc[client_index,'AMT_CREDIT'],
    'goods_price': df.loc[client_index,'AMT_GOODS_PRICE'],
    'annuity': df.loc[client_index,'AMT_ANNUITY'],
    'target': df.loc[client_index,'target'],
    'cluster': df.loc[client_index,'cluster'],
    'proba': df.loc[client_index,'proba'],
    'proba_bin':df.loc[client_index,'proba_group']
}

# Add a slider to the sidebar:
#add_slider = st.sidebar.slider('Select a client',)

filtered_df = df[df['SK_ID_CURR'] == select_client]

#--------------------------
# PAGE TITLE
#--------------------------
st.write('# **HOME CREDIT APPLICATION - CLIENT DETAILS**')


#--------------------------
# CLIENT DESCRIPTION
#--------------------------

with st.beta_container():
    col1, col2, col3 = st.beta_columns(3)

    with col1:
        st.write("## **Client Details**")
        st.markdown(f"**Gender:** {key_data['gender']}")
        st.markdown(f"**Age:** {key_data['age']:.0f} yo")
        st.markdown(f"**Nb children:** {key_data['nb_children']:.0f}")
        st.markdown(f"**Family Status:** {key_data['family_status']}")
        st.markdown(f"**Education Type:** {key_data['education_type']}")

    with col2:
        st.write("## **Occupation & Properties**")
        st.markdown(f"**Occupation Type:** {key_data['occupation_type']}")
        st.markdown(f"**Income Type:** {key_data['income_type']}")
        st.markdown(f"**Annual Income:** ${key_data['income_tot']:,.0f}")
        st.markdown(f"**Owns a car:** {key_data['owns_car']}")
        st.markdown(f"**Owns realty:** {key_data['owns_realty']}")

    with col3:
        st.markdown("## **Credit Application**")
        st.markdown(f"**Credit Amount:** ${key_data['credit_amount']:,.2f}")
        st.markdown(f"**Goods Price:** ${key_data['goods_price']:,.2f}")
        st.markdown(f"**Annuity:** ${key_data['annuity']:,.2f}")


#--------------------------
# CREDIT SCORE COMPARED TO OTHER CLIENTS
#--------------------------

st.write("## Where does the client stand?")
fig = barcharts(df, key_data)
st.write(fig)

#--------------------------
# CREDIT SCORE RESULT
#--------------------------


st.write('')
st.write('')
st.write('')
st.markdown(f"**Credit Score:** {key_data['proba']:,.2%}")
if key_data['target'] ==1:
    st.markdown("**Credit Application is likely to be approved**")
    st.balloons()
else: st.markdown("**Credit Application is unlikely to be approved**")


