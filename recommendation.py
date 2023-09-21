import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from clustering import predict_fraud

def load_pickle_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    
    return model

def get_feature_cols(product,cv,scaler_numeric_cols,numeric_cols,category_cols):
    #Convert names to vector
    vector_product=cv.transform(product.name).toarray()
    
    #Category cols to numeric
    category_cols_to_numeric=product[category_cols].values
    
    #Cluster col
    #cluster_col=product.cluster.values
    
    #Scale numeric cols
    scaled_product=scaler_numeric_cols.transform(product[numeric_cols])
    
    #combine vectorized name and scaled cols 
    #combined_product=np.c_[vector_product,scaled_product]
    
    combined_product=np.c_[vector_product,category_cols_to_numeric,scaled_product]
    
    return combined_product
    #return vector_product
    
    
def recommendation(product,df,cv,scaler_numeric_cols,numeric_cols,category_cols):
    product=product.reset_index(drop=True).copy()
    
    #if product.iloc[0,1]<3:
    #    product.iloc[0,1]=3
    #product.iloc[0,-1]=3
    
    if product.loc[0,'cluster']<3:
        product.loc[0,'cluster']=3
    
    product=get_feature_cols(product.iloc[0:1],cv,scaler_numeric_cols,numeric_cols,category_cols)
    similarity=pd.DataFrame([0.0]*df.shape[0],columns=['similarity_score'])
    
    step=10000
    for i in tqdm(range(0,df.shape[0]//step)):
        database_products=get_feature_cols(df.iloc[i*step:i*step+step],cv,scaler_numeric_cols,numeric_cols,category_cols)
        similarity.iloc[i*step:i*step+step]=cosine_similarity(product,database_products).T
        
    i+=i
    if i*step<df.shape[0]:
        database_products=get_feature_cols(df.iloc[i*step:])
        similarity.iloc[i*step:]=cosine_similarity(product,database_products).T

    return similarity.sort_values('similarity_score',ascending=False).iloc[1:]

def read_data(dataset="cleaned_amazon_products_with_cluster_without_duplicates.csv"):
    numeric_cols=['ratings', 'no_of_ratings', 'discount_price', 'actual_price', 'discount%']
    category_cols= ['main_category','sub_category']
    df=pd.read_csv(dataset,usecols=['name']+category_cols+numeric_cols+['cluster'])
    
    return df.copy()
    
def process_data(df):
    df.main_category=df.main_category.astype('category').cat.codes
    df.sub_category=df.sub_category.astype('category').cat.codes
    

def main():
    numeric_cols=['ratings', 'no_of_ratings', 'discount_price', 'actual_price', 'discount%']
    category_cols= ['main_category','sub_category']
    
    df=read_data()
    process_data(df)

    cv=load_pickle_model("count_vectorize_model.pkl")
    scaler_numeric_cols=load_pickle_model("recommendation_scalar_model.pkl")
    
    numeric_cols=['discount_price']
    
    selected_product=df.iloc[6144:6144+1]
    
    print("Fraud item :",predict_fraud(selected_product))
    
    similarity_df=recommendation(selected_product,df,cv,scaler_numeric_cols,numeric_cols,category_cols)
    
    new_df=read_data()
    print(new_df.iloc[similarity_df.index].head(20))
    
    
if __name__ == '__main__':
    main()
