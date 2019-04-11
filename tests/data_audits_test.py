# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:20:11 2019

@author: strq
"""
import pytest
from data_audits.data_audits import *

import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs

def create_test_data():
    # Unique Numbers
    arrNumbers = np.arange(0,5)
    # Numbers with Nan
    arrNanNumbers = np.array([1, np.nan, 2, np.nan, 3])
    # Non unique objects
    arrObj = ["A", "A", "B", "C", "D"]
    
    # Categorical
    # Categorical with Nan
    serCat = pd.Series(["a", "b", "c", "b", "a"])
    serNanCatl = pd.Series(["b", "a", "c", "e", "f"])
    cat_type = pd.api.types.CategoricalDtype(
            categories=["a", "b", "c", "d"], ordered=True)
    serCategorical = serCat.astype(cat_type)
    serNanCategorical = serNanCatl.astype(cat_type)
    
    serNoCat = pd.Series(["a", "b", "c", "b", "a"])
    cat_no_order_type = pd.api.types.CategoricalDtype(
            categories=["a", "b", "c", "d"], ordered=False)
    serNoOrderCategorical = serNoCat.astype(cat_no_order_type)
    
    # Outlier
    arrOutlier = np.array([1,1,1,1,10])
    
    dictionary = {"id": arrNumbers,
              "nanNumbers": arrNanNumbers,
              "nonUniqueObjects": arrObj,
              "categorical": serCategorical,
              "nanCategorical": serNanCategorical,
              "noOrderCategorical": serNoOrderCategorical,
              "sigOutlier": arrOutlier}
    
    dfData = pd.DataFrame(dictionary)
    dfData.insert(0,'TimeStamp',pd.datetime.now().replace(microsecond=0))
    dfData.loc[0,"TimeStamp"] = pd.Timestamp('2018-12-01 08:00:00.000000', tz=None)
    dfData.loc[1,"TimeStamp"] = pd.Timestamp('2018-12-01 08:00:01.000000', tz=None)
    dfData.loc[2,"TimeStamp"] = pd.Timestamp('2019-01-31 08:00:00.000000', tz=None)
    dfData.loc[3,"TimeStamp"] = pd.Timestamp('2019-01-31 08:01:00.000000', tz=None)
    dfData.loc[4,"TimeStamp"] = pd.Timestamp('2021-01-31 09:00:00.000000', tz=None)
    return dfData

def create_dctValuesAllValid():
    dctValuesAllValid = dict()
    dctValuesAllValid.update({"id":[0,4]})
    dctValuesAllValid.update({"nanNumbers":[1,3]})
    dctValuesAllValid.update({"nonUniqueObjects":["A", "B", "C", "D"]})
    dctValuesAllValid.update({"categorical":["a", "d"]})
    dctValuesAllValid.update({"nanCategorical":["a", "d"]})
    dctValuesAllValid.update({"noOrderCategorical":["a", "b", "c", "d"]})
    dctValuesAllValid.update({"sigOutlier":[1,10]})
    return dctValuesAllValid

def create_dctValuesNoneValid():
    dctValuesNoneValid = dict()
    dctValuesNoneValid.update({"id":[5,6]})
    dctValuesNoneValid.update({"nanNumbers":[4,5]})
    dctValuesNoneValid.update({"nonUniqueObjects":[]})
    dctValuesNoneValid.update({"categorical":[]})
    dctValuesNoneValid.update({"nanCategorical":[]})
    dctValuesNoneValid.update({"noOrderCategorical":[]})
    dctValuesNoneValid.update({"sigOutlier":[2,4]})
    return dctValuesNoneValid

def create_dctOutliers():
    dctOutliers = dict()
    dctOutliers.update({"id":[0,4]})
    dctOutliers.update({"nanNumbers":[1,3]})
    dctOutliers.update({"sigOutlier":[1,1]})
    return dctOutliers

def create_dctWrongOutliersDct():
    dctOutliers = dict()
    dctOutliers.update({"id":["A"]})
    dctOutliers.update({"nanNumbers":[1,3]})
    dctOutliers.update({"sigOutlier":[1,1]})
    return dctOutliers

def create_test_data_no_time():
    # Unique Numbers
    arrNumbers = np.arange(0,5)
    # Numbers with Nan
    arrNanNumbers = np.array([1, np.nan, 2, np.nan, 3])
    # Non unique objects
    arrObj = ["A", "A", "B", "C", "D"]
    
    # Categorical
    # Categorical with Nan
    serCat = pd.Series(["a", "b", "c", "b", "a"])
    serNanCatl = pd.Series(["b", "a", "c", "e", "f"])
    cat_type = pd.api.types.CategoricalDtype(
            categories=["a", "b", "c", "d"], ordered=True)
    serCategorical = serCat.astype(cat_type)
    serNanCategorical = serNanCatl.astype(cat_type)
    
    # Outlier
    arrOutlier = np.array([1,1,1,1,10])
    
    dictionary = {"id": arrNumbers,
                  "nanNumbers": arrNanNumbers,
                  "nonUniqueObjects": arrObj,
                  "categorical": serCategorical,
                  "nanCategorical": serNanCategorical,
                  "sigOutlier": arrOutlier}
    
    dfData = pd.DataFrame(dictionary)
    return dfData

class TestDataAudits(object):
    """
    def test_wrong_proportion_of_missing_values(self):
        with pytest.raises(ValueError):
            dfData = create_test_data()
            dfResult = proportion_of_missing_values(dfData, 
                                            strDetectionMode = "blub", 
                                            boolDropOutliers = False)
    """
      
    def test_data_type_mapper(self):
        dfData = create_test_data()
        serResults = data_type_mapper("Object")
        
    def test_data_description(self):
        dfData = create_test_data()
        dfResults = data_description(dfData)
        
    def test_classify_data_type_logic(self):
        dfData = create_test_data()
        dfResults = classify_data_type_logic(dfData["id"])
        
    def test_proportion_of_missing_values(self):
        dfData = create_test_data()
        serResults = proportion_of_missing_values(dfData)     
        assert serResults[5] == 0.4

    def test_proportion_of_invalid_values_All_Valid(self):
        dfData = create_test_data()
        dctValues = create_dctValuesAllValid()
        dfResults = proportion_of_invalid_values(dfData, dctValues)
        assert(dfResults.loc[1, "Proportion_Invalid_Values"] == 0.4)
  
    def test_proportion_of_invalid_values_None_Valid(self):
        dfData = create_test_data()    
        dctValues = create_dctValuesNoneValid()
        dfResults = proportion_of_invalid_values(dfData, dctValues)
        assert(dfResults.loc[1, "Proportion_Invalid_Values"] == 1)

    def test_proportion_of_outliers(self):
        dfData = create_test_data()
        dctOutliers = create_dctOutliers()
        dfResults = proportion_of_outliers(dfData, dctOutliers)
        assert(dfResults.loc[2, "Proportion_Outliers"] == 0.2)
    
    def test_wrong_types_list_proportion_of_outliers(self):
        with pytest.raises(AssertionError):
            dfData = create_test_data()
            dctOutliers = create_dctWrongOutliersDct()
            proportion_of_outliers(dfData, dctOutliers)
    
    def test_value_range_of_features(self):
        dfData = create_test_data()
        serResults = value_range_of_features(dfData)
        assert(serResults[0] == [0,4])

    def test_number_of_unique_values(self):
        dfData = create_test_data()
        serResults = number_of_unique_values(dfData)
        assert(serResults[0] == 5)

    def test_granularity_of_timestamp_feature(self):
        dfData = create_test_data()
        dfResults = granularity_of_timestamp_feature(dfData, ["D"])
        assert(dfResults.loc[0, "Maximum"] == 731.0409722222222)
        
    def test_granularity_of_timestamp_feature_wrong_timeConversion(self):
        with pytest.raises(TypeError):
            dfData = create_test_data()
            granularity_of_timestamp_feature(dfData, "E")

    def test_convert_time_column_and_granularity_of_timestamp(self):
        dfData = create_test_data()
        dfResults = convert_time_column_and_granularity_of_timestamp(
                dfData, ["TimeStamp"])
        
    def test_proj_kMeans(self):
        X, y_true = make_blobs(n_samples=300, centers=4,
                               cluster_std=0.60, random_state=0)
        dfX = pd.DataFrame(X)
        #dfY = pd.DataFrame(y_true)
        proj_kMeans(dfX,2,False)
        
    def test_kMeans(self):
        X, y_true = make_blobs(n_samples=300, centers=4,
                               cluster_std=0.60, random_state=0)
        dfX = pd.DataFrame(X)
        #dfY = pd.DataFrame(y_true)
        kMeans(dfX,2)
    
    def test_spectral_clustering(self):
        X, y_true = make_blobs(n_samples=300, centers=4,
                               cluster_std=0.60, random_state=0)
        dfX = pd.DataFrame(X)
        #dfY = pd.DataFrame(y_true)
        spectral_clustering(dfX,2)
"""        
    def test_wrong_statistical_moments_of_features(self):
        dfData = create_test_data()
        dfResult = statistical_moments_of_features(dfData)
        
    def test_wrong_create_a_correlation_matrix(self):
        dfData = create_test_data()
        dfResult = create_a_correlation_matrix(dfData)
        
    def test_visualize_a_correlation_matrix_heatmap(self):
        dfData = create_test_data()
        dfResult = visualize_a_correlation_matrix_heatmap(dfData)
        
    def test_wrong_visualize_a_correlation_matrix_scatter_plot_matrix(self):
        dfData = create_test_data()
        dfResult = visualize_a_correlation_matrix_scatter_plot_matrix(dfData)
        
    def test_wrong_cross_correlation(self):
        dfData = create_test_data()
        dfResult = cross_correlation(dfData)
        
    def test_wrong_distribution_of_feature_histogram(self):
        dfData = create_test_data()
        dfResult = cross_correlation(dfData)    
    
    def test_wrong_distribution_of_feature_box_plot(self):
        dfData = create_test_data()
        dfResult = cross_correlation(dfData)
    
    def test_wrong_distribution_of_feature_density_plot(self):
        dfData = create_test_data()
        dfResult = cross_correlation(dfData)
    
"""

#
#
#
#
#