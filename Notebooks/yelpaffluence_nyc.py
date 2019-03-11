import os
os.environ["PROJ_LIB"] = "/anaconda3/share/proj/"; # Location of 'epsg' in directories

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from yelpapi import YelpAPI

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

class YelpAffluence_NYC:
    """
    Class created January 2019 by Britt Allen, Bernard Kurka, Tom Ludlow in 
    General Assembly's Data Science Immersive as part of Project 4.  
    
    Object and methods allow user to identify a custom "affluence" metric on 
    a scale of $ to $$$$, similar to that used by Yelp for an establishment's 
    price rating.
    
    Methods fit a classification model of Yelp pricing data for New York City
    in January 2019.  **NOTE: Source data must be present in directory to function 
    properly.**  User may enter a string containing a ZIP code or neighborhood
    name, and the model will assign it an affluence rating relative to other
    neighborhoods/ZIPs in NYC.
    """
    
    def __init__(self, api_key=None):
        """
        Instantiate object with valid API key to access Yelp Fusion API.
        
        api_key: string
        """        
        # Instantiate variables
        self.api_key = api_key
        self.data = pd.read_csv('../Data/nyc_best.csv')
        
        self.prices = ['$','$\$','$\$\$','$\$\$\$']
        self.colors = ['tomato','goldenrod','seagreen','royalblue']
        self.cluster_num_dict = {0:0,1:3,2:1,3:2}
        self.cluster_dict = {0:'$',1:'$\$\$\$',2:'$\$',3:'$\$\$'}
        self.searched=False
        
        # Instantiate models
        self.features = ['pr_1s','pr_2s','pr_3s','pr_4s','pr_totms']
        self.kmeans = KMeans(n_clusters=4, init='random', algorithm='full', random_state=42, tol=0.0001)
        self.knn = KNeighborsClassifier(n_neighbors=15)
        self.ss = StandardScaler()
        self.result = pd.DataFrame(columns=['pred'])
        
    def query_to_df(self, loc_in, cat_in=['none'], 
                    sort_in='distance', limit_in=50, 
                    cols=['categories','alias','city','state','zip_code','price','review_count','latitude','longitude']):
        """Available arguments:
        loc_in (str): location (zip, city, neighborhood, etc.)
        cat_in (list): categories - default is ['restaurants','shopping','localservices']
        sort_in (str): sort criterion of 'distance','best_match','review_count' - default is 'distance'
        limit_in (int): number of results to pull per category, max is 50 - default is 50
        cols (list): columns for dataframe, matching API results key names - default is
        ['categories','alias','city','state','zip_code','price','review_count','latitude','longitude']
        """
        # Set Yelp Fusion API Key and establish API connection
        if not self.api_key:
            self.api_key = '21Pt2l8__qgIdL0ZpgYC_yWblJ_O8_vJ3_-tIybHDyuQl9oVBXAzAXQWqMmIrz7idLyc7owv4-lfSON0QjKJN4pvQei4rUQAGSZcGcVTQc4HtBseUcztUPkVrAItXHYx'
        api_obj = YelpAPI(self.api_key, timeout_s=3.0)

        # Instantiate empty DataFrame with desired output columns
        output_df = pd.DataFrame(columns=['search_term']+cols)

        # Create iterable list of limit amounts <= 50 so that full limit argument is covered
        # ex. 70 -> [50,20]
        limit_list = []
        if limit_in > 50:
            req = limit_in  # req starts at limit argument and counts down by 50 until < 50
            while req > 50:
                limit_list.append(50)
                req -= 50
            limit_list.append(req)
        else:
            limit_list.append(limit_in) # if req < 50 append remaining amount to list

        # Loop through category argument list items
        for cat in cat_in:
            cat_df = pd.DataFrame(columns=['search_term']+cols) # Create empty DataFrame with addl col for category
            for j, limit in enumerate(limit_list): # Perform API pulls with all limits in limit_list

                # API call saved to json dict
                if cat=='none':
                    response = api_obj.search_query(location=loc_in, sort_by=sort_in, limit=limit, offset=(j*50))
                else:
                    response = api_obj.search_query(location=loc_in, categories=[cat], sort_by=sort_in, limit=limit, offset=(j*50))
                response_df = pd.DataFrame(response['businesses']) # Save business data to DataFrame

                # Create iteration DataFrame to process each API response (up to 50 results)
                iter_df = pd.DataFrame(columns=['search_term']+cols)
                iter_df['search_term'] = [cat for i in range(len(response_df))] # Add category value for each row

                # Iterate through each requested column argument and format for storage in output DataFrame
                for col_name in cols:
                    # Convert list of categories into single comma-separated string
                    if col_name == 'categories':
                        # Exception handling: not all responses include all categories
                        try:
                            for k, cell in enumerate(response_df['categories']):
                                iter_cat_str = ''
                                for d in cell:
                                    iter_cat_str += str(d['alias']+', ')
                                iter_df.loc[k, 'categories'] = iter_cat_str[:-2] # Save final string, without final ', ' 
                        except:
                            pass
                    elif col_name in ('city','state','zip_code'): # Access location data through 'location' key value
                        try:
                            iter_df[col_name] = [response_df['location'][i][col_name] for i in range(response_df.shape[0])]
                        except:
                            pass
                    elif col_name in ('latitude','longitude'): # Access latitude/longitude through 'coordinates' key value
                        try:
                            iter_df[col_name] = [response_df['coordinates'][i][col_name] for i in range(response_df.shape[0])]
                        except:
                            pass
                    else:
                        try:
                            iter_df[col_name] = response_df[col_name] # Anything else access directly
                        except:
                            pass
                cat_df = cat_df.append(iter_df)
            output_df = output_df.append(cat_df)
        output_df.index = range(output_df.shape[0])
        return output_df
    
    def api_pull(self, zip_list, cats=[None], sort='best_match', limit=50):
        column_list = ['zip','city','state','cat',
                       'pr_1','rv_1','pr_2','rv_2',
                       'pr_3','rv_3','pr_4','rv_4',
                       'avg_lat','avg_long']

        api_data = pd.DataFrame(columns=column_list)
        for z in zip_list:
            print(z)
            df = self.query_to_df(z, cats, limit_in=limit, sort_in=sort)

            loop_df = pd.Series(index=column_list)

            loop_df['zip'] = z
            try:
                loop_df['city'] = df.city.value_counts(ascending=False).index[0]
                loop_df['state'] = df.state.value_counts(ascending=False).index[0]
            except: 
                pass
            
            loop_df['cat'] = 'None'

            loop_df['pr_1'] = df[df.price=='$'].shape[0]
            loop_df['rv_1'] = df[df.price=='$'].review_count.sum()
            loop_df['pr_2'] = df[df.price=='$$'].shape[0]
            loop_df['rv_2'] = df[df.price=='$$'].review_count.sum()
            loop_df['pr_3'] = df[df.price=='$$$'].shape[0]
            loop_df['rv_3'] = df[df.price=='$$$'].review_count.sum()
            loop_df['pr_4'] = df[df.price=='$$$$'].shape[0]
            loop_df['rv_4'] = df[df.price=='$$$$'].review_count.sum()

            loop_df['avg_lat'] = df.latitude.mean()
            loop_df['avg_long'] = df.longitude.mean()

            api_data = api_data.append(loop_df, ignore_index=True)

        api_data.zip = api_data.zip.astype(str)

        return api_data
    
    def fit_model(self):
        """
        Create sklearn ensemble models and fit based on yelp_nyc_total.csv data
        """ 
        self.data['pr_2m'] = self.data['pr_2'] * 2
        self.data['pr_3m'] = self.data['pr_3'] * 3
        self.data['pr_4m'] = self.data['pr_4'] * 4
        self.data['pr_totm'] = self.data['pr_1'] + self.data['pr_2m'] + self.data['pr_3m'] + self.data['pr_4m']
        
        ss_feat = ['pr_1','pr_2','pr_3','pr_4','pr_totm']
        data_ss = self.ss.fit_transform(self.data[ss_feat])
        
        self.data['pr_1s'] = [data_ss[i][0] for i in range(len(data_ss))]
        self.data['pr_2s'] = [data_ss[i][1] for i in range(len(data_ss))]
        self.data['pr_3s'] = [data_ss[i][2] for i in range(len(data_ss))]
        self.data['pr_4s'] = [data_ss[i][3] for i in range(len(data_ss))]
        self.data['pr_totms'] = [data_ss[i][4] for i in range(len(data_ss))]
        
        # self.data[ss_feat] = data_ss
        
        self.kmeans.fit(self.data[self.features])
        self.data['pred'] = self.kmeans.labels_
        self.data['pred_$'] = self.data.pred.map(lambda x: self.cluster_dict[x])
        self.data['pred_num'] = self.data.pred.map(lambda x: self.cluster_num_dict[x])
        self.knn.fit(self.data[self.features], self.data.pred)
        
    def get_affluence(self, locations):
        """
        Return price rating ($, $$, $$$, $$$$) for a NYC ZIP or 
        neighborhood search term
        
        location: string, list of strings
        """
        self.result = self.api_pull(locations, sort='best_match', limit=100)
        
        self.result['pr_2m'] = self.result['pr_2'] * 2
        self.result['pr_3m'] = self.result['pr_3'] * 3
        self.result['pr_4m'] = self.result['pr_4'] * 4
        self.result['pr_totm'] = self.result['pr_1'] + self.result['pr_2m'] + self.result['pr_3m'] + self.result['pr_4m']
        
        ss_feat = ['pr_1','pr_2','pr_3','pr_4','pr_totm']
        result_ss = self.ss.transform(self.result[ss_feat])
        
        self.result['pr_1s'] = [result_ss[i][0] for i in range(len(result_ss))]
        self.result['pr_2s'] = [result_ss[i][1] for i in range(len(result_ss))]
        self.result['pr_3s'] = [result_ss[i][2] for i in range(len(result_ss))]
        self.result['pr_4s'] = [result_ss[i][3] for i in range(len(result_ss))]
        self.result['pr_totms'] = [result_ss[i][4] for i in range(len(result_ss))]
            
        # self.result[self.features] = result_ss
        
        self.result['pred'] = self.knn.predict(self.result[self.features])
        self.result['pred_$'] = self.result.pred.map(lambda x: self.cluster_dict[x])
        self.result['pred_num'] = self.result.pred.map(lambda x: self.cluster_num_dict[x])
        
        self.result.zip = self.result.zip.str.split('.', expand=True)[0]
        self.result.drop(['cat'], axis=1, inplace=True)
        self.result = self.result.rename(columns={'zip':'location'})
        
        self.result = self.result.sort_values('pred_$')
        self.result.index = range(self.result.shape[0])
        
        self.searched=True
        
        return self.result
        
    def plot_query_results_mean(self):
        if not self.searched:
            return 'No stored results.'
        result_prices = self.result.pred_num.unique()
        n_tall = 2
        if len(result_prices)<3:
            n_tall = 1
        hide_last = False
        if len(result_prices) % 2:
            hide_last = True
        fig, ax = plt.subplots(n_tall,2,figsize=(12,6*n_tall), sharey=True)
        ax = ax.ravel()

        for i, n in enumerate(result_prices):
            ax[i].bar(['$','$\$','$\$\$','$\$\$\$'],
                  [self.result[self.result.pred_num==n].pr_1.mean(),self.result[self.result.pred_num==n].pr_2.mean(),
                   self.result[self.result.pred_num==n].pr_3.mean(),self.result[self.result.pred_num==n].pr_4.mean()],
                  label='Results', color=self.colors[n])
            ax[i].set_title('\$'*(n+1), fontsize=18)
            ax[i].set_xlabel('Yelp Price Ratings', fontsize=14)
            ax[i].set_ylabel('Mean Number of Businesses', fontsize=14)
            ax[i].grid()
            if hide_last:
                ax[len(result_prices)].set_visible(False)
        fig.suptitle('Mean Results for Queried Locations', fontsize=22)
        #plt.show();
        
    def plot_query_results(self):
        if not self.searched:
            return 'No stored results.'
        n_results = self.result.shape[0]
        n_tall = int(n_results / 2)
        hide_last = False
        if n_results % 2: 
            n_tall += 1
            hide_last = True
        fig, ax = plt.subplots(n_tall,2,figsize=(12,n_tall*6), sharey=True)
        ax = ax.ravel()

        for i in range(n_results):
            ax[i].bar(['$','$\$','$\$\$','$\$\$\$'],
                  [self.result.loc[i, 'pr_1'],self.result.loc[i, 'pr_2'],
                   self.result.loc[i, 'pr_3'],self.result.loc[i, 'pr_4']],
                  label=self.result.loc[i,'location'], color=self.colors[self.cluster_num_dict[self.result.loc[i,'pred']]])
            ax[i].set_title(self.result.loc[i, 'location'].title()+': '+self.result.loc[i,'pred_$'], fontsize=16)
            ax[i].set_xlabel('Yelp Price Ratings', fontsize=12)
            ax[i].set_ylabel('Number of Businesses in Location', fontsize=12)
            ax[i].grid()
            
        if hide_last:
            ax[n_results].set_visible(False)
            
        fig.suptitle('Individual Query Results', fontsize=22)
        #plt.show();
    
    def plot_model_results(self):
        """
        Display Matplotlib Pyplot results for neighborhood metrics
        """
        fig, ax = plt.subplots(2,2,figsize=(12,12), sharey=True)
        ax = ax.ravel()

        for i in range(4):
            ax[i].bar(['$','$\$','$\$\$','$\$\$\$'],
                  [self.data[self.data.pred_num==i].pr_1.mean(),self.data[self.data.pred_num==i].pr_2.mean(),
                   self.data[self.data.pred_num==i].pr_3.mean(),self.data[self.data.pred_num==i].pr_4.mean()],
                  label=self.prices[i], color=self.colors[i])
            ax[i].set_title('\$'*(i+1), fontsize=18)
            ax[i].set_xlabel('Yelp Price Ratings', fontsize=14)
            ax[i].set_ylabel('Mean Number of Businesses', fontsize=14)
            ax[i].grid()
        
        fig.suptitle('Mean Results for All NYC ZIPs', fontsize=22)
        #plt.show();
        
    def plot_nyc_map(self, x_in=16, y_in=16):
        """
        Display Matplotlib Pyplot results using latitude/longitude
        """
        plt.figure(figsize=(x_in,y_in))
        plt.clf()
        m = Basemap(projection='cyl', llcrnrlat=40.5, urcrnrlat=40.92, 
                    llcrnrlon=-74.3, urcrnrlon=-73.7, resolution='h')
        m.fillcontinents(color="#eeeeee", alpha=0.2)
        m.drawcoastlines()
        m.drawmapboundary()

        for i in range(4):
            plt.scatter(self.data[self.data.pred_num==i].avg_long, 
                        self.data[self.data.pred_num==i].avg_lat, 
                        label=self.prices[i], color=self.colors[i], alpha=0.8, s=20)
        adj_lat = 0
        self.result['loc_len'] = [len(self.result.location[i]) for i in range(len(self.result))]
        len_sort = self.result.sort_values('loc_len', ascending=False)
        
        len_sort.index = range(len(len_sort))
        for j in range(len_sort.shape[0]):
            try:
                plt.scatter(len_sort.loc[j,'avg_long'], len_sort.loc[j,'avg_lat'], 
                            marker='d', label=None, 
                            color=self.colors[self.cluster_num_dict[len_sort.loc[j,'pred']]], s=200)
                plt.annotate(len_sort.loc[j,'location'].title(), 
                             xy=(len_sort.loc[j,'avg_long'],len_sort.loc[j,'avg_lat']),
                             xytext=(-74.28,40.85-adj_lat),
                             fontsize=16, color=self.colors[self.cluster_num_dict[len_sort.loc[j,'pred']]],
                             arrowprops={'arrowstyle':'-', 
                                         'color':self.colors[self.cluster_num_dict[len_sort.loc[j,'pred']]]})
                adj_lat += 0.02
            except: pass

        plt.title('NYC by Yelp ZIP Code', fontsize=22)
        plt.xlabel('Longitude', fontsize=14)
        plt.xticks(np.arange(-74.3,-73.65,.1))
        plt.ylabel('Latitude', fontsize=14)
        plt.yticks(np.arange(40.5,40.95,.05))
        plt.legend(loc=2, fontsize='xx-large')
        #plt.show();
        
    def filter_zips(self, pr_val):
        return self.data[self.data.pred_num==pr_val]