
# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
from functions import barcharts, gaugechart, load_data, comparisonchart

#--------------------------
# ICON & TITLE
#--------------------------

st.set_page_config(page_title="Home Credit - Client Scoring",
                   page_icon=":money_with_wings:",
                   layout='wide')

#--------------------------
# VARIABLES & DATA PREP
#--------------------------

# Will only run once if already cached

df = load_data('customers_data.csv')

# Add a selectbox to the sidebar:

st.sidebar.header('Filter clients :')

age_category_list = ['All'] + np.unique(df['age_group']).tolist()
age_category = st.sidebar.selectbox("Age Group", age_category_list)

gender_list = ['All'] + np.unique(df['CODE_GENDER']).tolist()
gender = st.sidebar.selectbox("Gender", gender_list)

ed_level_list = ['All'] + np.unique(df['NAME_EDUCATION_TYPE']).tolist()
ed_level = st.sidebar.selectbox("Education Level", ed_level_list)

# Saving selections in a global filter

if (age_category != 'All'):
    age_filter = (df['age_group'] == age_category)
else: age_filter = (df['age_group'].isin(age_category_list))

if (ed_level != 'All'):
    ed_filter = (df['NAME_EDUCATION_TYPE'] == ed_level)
else: ed_filter = (df['NAME_EDUCATION_TYPE'].isin(ed_level_list))

if (gender != 'All'):
    gender_filter = (df['CODE_GENDER'] == gender)
else: gender_filter = (df['CODE_GENDER'].isin(gender_list))

client_list = df[age_filter & ed_filter & gender_filter]["SK_ID_CURR"].tolist()

st.sidebar.markdown("-------------------------------------------------")

st.sidebar.header('Select a client :')
select_client = st.sidebar.selectbox('Choose client ID',client_list)

client_index = df[df["SK_ID_CURR"] == select_client].index[0]

key_data = {
    'client_id': df.loc[client_index,'SK_ID_CURR'],
    'gender': df.loc[client_index,'CODE_GENDER'],
    'age': df.loc[client_index,'age'],
    'age_group': df.loc[client_index, 'age_group'],
    'debt_ratio': df.loc[client_index, 'debt_ratio'],
    'debt_group': df.loc[client_index, 'debt_group'],
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


filtered_df = df[df['SK_ID_CURR'] == select_client]

#--------------------------
# PAGE TITLE
#--------------------------
st.write('# **HOME CREDIT APPLICATION - RISK EVALUATION**')
st.write('')
st.write('')
st.write('')

st.markdown("-------------------------------------------------")

#--------------------------
# CLIENT DESCRIPTION
#--------------------------

with st.beta_container():
    col1, col2, col3 = st.beta_columns(3)

    with col1:
        st.write("## **Client Details**")
        st.markdown(f"**Gender:** {key_data['gender']}")
        st.markdown(f"**Age:** {key_data['age']:.0f} yo")
        st.markdown(f"**Nb children:   ** {key_data['nb_children']:.0f}")
        st.markdown(f"**Family Status:   ** {key_data['family_status']}")
        st.markdown(f"**Education Type:   ** {key_data['education_type']}")

    with col2:
        st.write("## **Occupation & Properties**")
        st.markdown(f"**Occupation Type:   ** {key_data['occupation_type']}")
        st.markdown(f"**Income Type:   ** {key_data['income_type']}")
        st.markdown(f"**Annual Income:   ** ${key_data['income_tot']:,.0f}")
        st.markdown(f"**Owns a car:   ** {key_data['owns_car']}")
        st.markdown(f"**Owns realty:   ** {key_data['owns_realty']}")

    with col3:
        st.markdown("## **Credit Application**")
        st.markdown(f"**Credit Amount:   ** ${key_data['credit_amount']:,.2f}")
        st.markdown(f"**Goods Price:   ** ${key_data['goods_price']:,.2f}")
        st.markdown(f"**Annuity:   ** ${key_data['annuity']:,.2f}")

st.markdown("-------------------------------------------------")

#--------------------------
# CREDIT SCORE RESULT
#--------------------------

st.write('')
st.write('')
st.write('')
st.write('')


with st.beta_container():
    cola, colb = st.beta_columns(2)

    with cola:
        st.write('## **Credit Risk Value :**',unsafe_allow_html=True)
        #st.markdown(f"**Credit Score:** {key_data['proba']:,.2%}")
        fig2 = gaugechart(key_data)
        st.write(fig2)

    with colb:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        if key_data['target'] ==1:
            st.markdown("## Credit Application is **risky** & likely to **default**")
        else:
            st.markdown("## Credit Application is **safe** & likely to be **approved**")
            st.balloons()

#--------------------------
# CREDIT SCORE COMPARED TO OTHER CLIENTS
#--------------------------

st.markdown("-------------------------------------------------")


st.write("## **Where does the client stand?**")
st.write("*Comparison is made with clients filtered using the left panel*")
fig = barcharts(df[df["SK_ID_CURR"].isin(client_list)], key_data)
st.write(fig)

#--------------------------
# CREDIT SCORE COMPARED TO OTHER CLIENTS
#--------------------------

st.markdown("-------------------------------------------------")


st.write('')
st.write('## **Score comparison with similar clients :** ')
st.write("*Comparison is made with clients filtered using the left panel*")
st.write('')

fig3 = comparisonchart(df[df["SK_ID_CURR"].isin(client_list)],key_data)
st.write(fig3)


