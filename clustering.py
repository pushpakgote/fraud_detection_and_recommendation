import pandas as pd
import numpy as np
import pickle

def load_clustering_model(kmeans_model="kmeans_model.pkl"):
    with open(kmeans_model, "rb") as f:
        kmeans_model = pickle.load(f)
    
    return kmeans_model

def load_scaling_model(standard_scaler_model="standard_scaler_model.pkl"):
    with open(standard_scaler_model, "rb") as f:
        standard_scaler_model = pickle.load(f)
        
    return standard_scaler_model

def load_pickle_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_cluster(X,kmeans_model,standard_scaler_model):
    
    clustering_cols=['ratings', 'discount%']
    
    #X can be pandas dataframe or numpy array , converting 1d to 2d and if X is numpy then numpy -> pandas
    if X.ndim == 1:
        if isinstance(X,pd.core.series.Series):
            X=pd.DataFrame(X.values.reshape(1,-1),columns=clustering_cols)
            
        elif isinstance(X,np.ndarray):
            X=pd.DataFrame(X.reshape(1,-1),columns=clustering_cols)
            
    elif X.ndim==2 and isinstance(X,np.ndarray):
        X=pd.DataFrame(X,columns=clustering_cols)
        
    X_scaled=pd.DataFrame(standard_scaler_model.transform(X),columns=clustering_cols)
    
    kmeans_result=kmeans_model.predict(X_scaled)
    
    #Redifining clusters to make recommendations better
    # cluster 0 -> 4 means worst to best products
    clusters_redefined={1:0,
      3:1,
      2:2,
      0:3}
    
    kmeans_result= np.array( list(map(clusters_redefined.get,kmeans_result)) )
    X['cluster'] = kmeans_result
    
    return kmeans_result,X

def predict_fraud(X):
     return ((X.cluster==0) & (X.ratings<6) & (X['discount%']>70)) | ((X.cluster==2) & (X.ratings<3) & (X['discount%']>75))
    

def main():
    kmeans_model=load_pickle_model("kmeans_model.pkl")
    standard_scaler_model=load_pickle_model("standard_scaler_model.pkl")
    
    #Get centroids
    #cluster_centroids=pd.DataFrame(standard_scaler_model.inverse_transform(kmeans_model.cluster_centers_),   columns=kmeans_model.feature_names_in_)
    
    clusters,x=predict_cluster(np.array([2.6,83]),kmeans_model,standard_scaler_model)
    print(predict_fraud(x))
    
if __name__ == "__main__":
    main()