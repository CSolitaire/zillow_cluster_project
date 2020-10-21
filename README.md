# Zillow Clustering Project

### Author: Corey Solitaire

## Project Goals: 

1. Explore features of Zillow data set and create 3 clusters that are statistically significant in their relationship to logerror.
2. Create features from data clusters (or cluster centers as centroids) and use them to train a regressio model to predict logerror.
3. Determine the drivers of logerror in Zillows Zestimate.
4. Develop and deliver a professional jupyter notebook outlineing your experience and familiarity with the process of data clustering, module creation, and technical delivery.
5. Develop and deliver a professional jupyter notebook outlineing your experience and familiarity with the process of data clustering, module creation, and technical delivery¶ 

## Executive Summary: 

**Project Summary:**   
The purpose to this project was to access single unit properties sold in 2007, combine features of those properties in to clusters (feature engeneering) and then develop models based on those features that would best predict logerror (log(Zestimate)−log(SalePrice)).

**Background:**   
The Zillow dataset (a competition dataset from Kaggle) consists of information on 2.9 million real estate properties in three counties (Los Angeles, Orange and Ventura, California) sold between October 2016 to December 2017. This project foucuses on the LA County dataset, the largest and most complex of the three.

**Process:**   
3 clusters (room, price, location) were created using the training dataset and 5 of these engeneered features were utilized to build a predictive linear model. Baseline was estabolished using the mean of the y-train based on the distribution of the data. Out of several regression model the linear regression model preformed best on train and validate data sets, however it failed to beat baseline on the test data set.

**Results and Conclusions:**   
Modeling produced an average RMSE of 0.17%, however that error represents a -6.63% improvement over baseline. Model accuracy did not significantly change over train, validate, test suggesting a need for further feature engeneering (cluster design) or different modeling techniques. There exists the possibility that regression models may not be the best tool to predict logerror in this dataset.

**Reccomendations:**   
    Too Much Data?:Investigate impact of further reducing outliers with regards to logerror.
    Best Features?: reexamine features to produce new clusters that are better able to predict logerror.
    New Model?: Regression models might not the best approch to predict logerror.
    Other Counties?: This model may work better on other CA counties, further investigation is necessary.

## Instructions for Replication

Files are located in Git Repo [here](https://github.com/CSolitaire/zillow_cluster_project)
User will need env.py file with access to Codeup database 

## Data Dictionary

  ---                            ---
| **Feature**                  | **Definition**                                                |
| ---                          | ---                                                           |
| bathroomcnt                  | Number of bathrooms in home                                   |
| LA                           | Dummy variable County                                         |
| calculatedfinishedsquarefeet | Total home square footage                                     |
| Orange                       | Dummy variable County                                         |
| Ventura                      | Dummy variable County                                         |
| age                          | 2017 - 'yearbuilt'                                            |
| taxrate                      | 'taxamount'/'taxvaluedollarcnt'                               |
| latitude                     | Angular distance north or south of the earth's equator        |
| longitude                    | Geographic coordinate, east–west position on the Earth        |
| acres                        | lotsizesquarefeet' / 43560                                    |
| structure_dollar_per_sqft    | 'structurevaluetaxdollarcnt' / 'calculatedfinishedsquarefeet' |
| land_dollar_per_sqft         | 'taxvaluedolarcnt' / 'lotsizesquarefeet'                      |
| bed_bath_ratio               | 'bedroomcnt' / 'bathroomcnt'                                  |
  ---                            ---                                                    
| **Target**                   | **Definition**                                                |
| ---                          | ---                                                           |
| logerror                     | Tax value in dollars                                          |

## Audience

- Zillow Data Science Team 

## Setting

- Professional

## Workflow

![](https://github.com/CSolitaire/zillow_cluster_project/blob/main/pipeline%20copy.jpg)