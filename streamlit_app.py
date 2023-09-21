# Import libraries
import streamlit as st
import pandas as pd
import os
import pickle

from recommendation import read_data,process_data,recommendation
from clustering import predict_fraud

import zipfile

@st.cache_data
def unzip():
    with zipfile.ZipFile("cleaned_amazon_products_with_cluster_without_duplicates.zip", 'r') as zip_ref:
        zip_ref.extractall()


path=os.getcwd()
rename_columns={'name':'Name',
                'main_category':'Category',
                'sub_category':'Sub Category',
                'ratings':'Ratings',
                'no_of_ratings':'Total Ratings',
                'discount_price':'Discount Price',
                'actual_price':'Actual Price',
                'discount%':'Discount %'}
show_cols=['name','main_category','sub_category','ratings','no_of_ratings','discount_price','actual_price','discount%']

@st.cache_data
def load_pickle_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    
    return model

def dataframe_with_selections(df):
    #df_with_selections = df[show_cols].rename(columns=rename_columns).copy()
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        #df_with_selections[show_cols].rename(columns=rename_columns),
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
       
        disabled=df.columns,
        height=300
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)


def recommend_and_detect_fraud_item(df,index,filename):
    numeric_cols=['ratings', 'no_of_ratings', 'discount_price', 'actual_price', 'discount%']
    category_cols= ['main_category','sub_category']
    
    #df=read_data()
    process_data(df)

    cv=load_pickle_model(os.path.join(path,"pickle_models","count_vectorize_model.pkl"))
    scaler_numeric_cols=load_pickle_model(os.path.join(path,"pickle_models","recommendation_scalar_model_with_cluster.pkl"))
    
    #numeric_cols=['discount_price']
    numeric_cols=['discount_price','cluster']
    
    selected_product=df.iloc[index:index+1]
    #print("Selected Product ndims =",selected_product.ndim)
    #print("Index:",selected_product.index)
    
    
    similarity_df=recommendation(selected_product,df,cv,scaler_numeric_cols,numeric_cols,category_cols)
    
    new_df=read_data(filename)

    #removing fraud items from recommendation
    fraud_filter= ~ predict_fraud( new_df.iloc[similarity_df.index].head(20))

    #print( fraud_filter )
    #print( fraud_filter[fraud_filter]  )
    #print( fraud_filter[fraud_filter].index )

    
    #st.dataframe(new_df.iloc[fraud_filter[fraud_filter].index].head(10))
    st.dataframe(new_df.iloc[fraud_filter[fraud_filter].index][show_cols].rename(columns=rename_columns).head(10))


    


# Page setup
st.set_page_config(page_title="Fraudulent Product Detection and Recommendation", layout="wide")

with st.sidebar:
    st.image("recommended_img.png",width=300)
    st.markdown("<h1 style='text-align: center;'>Fraudulent Product Detection and Recommendation</h1>", unsafe_allow_html=True)
    st.info("This webapp checks if product is suspicious and recommends better products.")
    st.info("For full code, head on to [@pushpakgote/fraud_detection_and_recommendation](https://github.com/pushpakgote/fraud_detection_and_recommendation)")
    
st.title("Fraudulent Product Detection and Recommendation")


filename="cleaned_amazon_products_with_cluster_without_duplicates.csv"

if filename not in os.listdir(path):
    unzip()

#Read file
df=read_data(filename)

# Use a text_input to get the keywords to filter the dataframe
text_search = st.text_input("Search any product", value="")

# Filter the dataframe using masks
m1 = df["name"].str.contains(text_search)
df_search = df[m1]



if text_search:
    st.caption("Select any one product")
    #selection = dataframe_with_selections(df_search)
    selection = dataframe_with_selections(df_search[show_cols].rename(columns=rename_columns))
    #print("selection ndims =",selection.ndim)

    if not selection.empty:
        
        index=selection.index[-1]
        fraud=predict_fraud(df.iloc[index])

        if fraud:
            st.header(":red[Suspicious product] :x:")
        else:
            st.header(":green[Product is safe] :heavy_check_mark:")

        #st.write(selection.iloc[-1:][show_cols].rename(columns=rename_columns))
        st.write(selection.iloc[-1:])

        # print(selection.index)
        # print(index)

        
        st.header("Recommendations")
        with st.spinner("Please wait ..."):
            recommend_and_detect_fraud_item(df,index,filename)


