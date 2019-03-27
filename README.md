# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Estimating Neighborhood Affluence With Yelp Data

## Project Authors
- Bernard Kurka | [LinkedIn](https://www.linkedin.com/in/bernardkurka) | [Email](mailto:bkexcel2014@gmail.com)
- Thomas Ludlow | [LinkedIn](https://www.linkedin.com/in/twludlow) | [Email](mailto:tludlow@gmail.com)
- Brittany Allen | [LinkedIn](https://www.linkedin.com/in/brittlallen) | [Email](mailto:thebrittallen@gmail.com)

## Presentation Slides
- https://docs.google.com/presentation/d/1a9jGkPlWrMeTOaTl-t5zsYTpBsu5WPCP6Zb0eF-3wws/edit?usp=sharing
- [NLT Project_ Estimating Neighborhood Affluence with Yelp Data — Bernard K, Thomas L, Brittany A.pdf](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/Project%204%20Ludlow_Allen_Kurka%20-%20Yelp%20Slides.pdf)


## Notebooks
- [Notebook 01 - Yelp Affluence Demo](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/01%20-%20Yelp%20Affluence%20Demo%20190118.ipynb)
- [Notebook 02 - Yelp Fusion API](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/02%20-%20Yelp%20Fusion%20API.ipynb)
- [Notebook 03 - IRS Data](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/03%20-%20IRS_data_collection_and_cleaning.ipynb)
- [Notebook 04 - API Pull for Training Data](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/04%20-%20API%20Pull%20for%20Training%20Data.ipynb)
- [Notebook 05 - Yelp Cluster Gridsearch](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/05%20-%20Yelp%20Cluster%20Grid%20Search%20-%20NYC.ipynb)
- [Notebook 06 - K-Means Modeling](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/06%20-%20Yelp%20K-Means%20Model%20-%20NYC.ipynb)
- [Notebook 07 - Yelp Class Development](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/07%20-%20Yelp%20Class%20Development.ipynb)


## Table of contents
- [Executive Summary](#Executive-Summary)
- [The Data](#The-Data)
- [Analysis](#Analysis)
- [Conclusion](#Conclusion)


### Executive Summary
[New Light Technologies (NLT)](https://www.linkedin.com/company/new-light-technologies) requested an economic analysis utilizing Yelp cost estimates ($, $$, $$$, $$$$) to estimate neighborhood affluence. In thinking about conventional sources of economic data like the U.S. Census Bureau or the IRS, two major flaws we can identify are 1) the reporting lags in acquiring this data, and 2) the ability to parse the data in varied ways. How can we develop a tool leveraging more agile data sources (like Yelp) that will help us measure local economic activity?

Our initial approach was to build a supervised learning, linear regression model, but we found that defining a proper affluence metric by using IRS zipcode-level AGI (adjusted gross income) was difficult to do. While traditional methods typically estimate wealth of a locality based on demographic characteristics like income, the novelty of our latter approach — conducting an unsupervised learning analysis with clustering algorithms — is in its use of big data related to commercial activity and cost of product and services as an indicator of affluence.

By counting the number of businesses in each price range ($, $$, $$$, $$$$) in Yelp's result page, we were able to use K-Means Clustering Model create 4 distinct clusters:
 - $ cluster: Limited activity across all price ranges ($, $$, $$$, $$$$)
 - $$ cluster: Moderate activity in $ and $$. Limited activity in $$$ and $$$$.
 - $$$ cluster: Highest activity in $$. Increased activity in $$$ and $$$$
 - $$$$ cluster: Highest activity in $$$ and $$$$. High activity in $$.
 
We also developed a Python Class library that, given a list of NYC Zipcodes or neighborhood names, will access the Yelp API to gather data for these neighborhoods, and use a KNN model to predict which cluster best contains it. The Class also has plotting functions to compare the given Zipcodes with the 4 static clusters, and there is also a option to print results on NYC map.

### The Data
##### Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**zip**|*object*|KMeans Clustering|String for each of 278 ZIP codes across New York City boroughs|
|**location**|*object*|KNN Classifier|String of ZIP code or text location query parameter being classified with KNN model|
|**pr_1s**|*float*|KMeans Clustering, KNN Classifier|Floating point standardized ("s") value for number of \$ priced ("pr_1") establishments in top 100 best match for search location|
|**pr_2ws**|*float*|KMeans Clustering, KNN Classifier|Floating point standardized ("s") value for number of \$\$ priced ("pr_2") establishments in top 100 best match for search location, weighted ("w") by price level|
|**pr_3ws**|*float*|KMeans Clustering, KNN Classifier|Floating point standardized ("s") value for number of \$\$\$ priced ("pr_3") establishments in top 100 best match for search location, weighted ("w") by price level|
|**pr_4ws**|*float*|KMeans Clustering, KNN Classifier|Floating point standardized ("s") value for number of \$\$\$\$ priced ("pr_4") establishments in top 100 best match for search location, weighted ("w") by price level|
|**pr_totws**|*float*|KMeans Clustering, KNN Classifier|Floating point standardized ("s") value for total weighted price counts ("tot", "w") for top 100 best match for search location|

- [Notebook 02 - Yelp Fusion API](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/02%20-%20Yelp%20Fusion%20API.ipynb)
- [Notebook 03 - IRS Data](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/03%20-%20IRS_data_collection_and_cleaning.ipynb)
- [Notebook 04 - API Pull for Training Data](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/04%20-%20API%20Pull%20for%20Training%20Data.ipynb)

Our data was acquired via Yelp's Fusion API `from yelpapi import YelpAPI`. We set our Yelp Fusion API Key and established an API connection. From there we stored the data we gathered into a `Pandas` DataFrame.
The search criteria included all Yelp's categories (e.g., shops, restaurants), filtered by Yelp's "best match" option.  We gathered the top 100 business prices and reviews from 278 NYC zipcodes. 

 
### Analysis
- [Notebook 05 - Yelp Cluster Gridsearch](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/05%20-%20Yelp%20Cluster%20Grid%20Search%20-%20NYC.ipynb)
- [Notebook 06 - K-Means Modeling](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/06%20-%20Yelp%20K-Means%20Model%20-%20NYC.ipynb)

In our analysis, we used a Grid search through 3 different clustering models (K-Means, Agglomerative and Hierarchical) and changing hyperparameters (e.g., n_clusters, inits, linkage_method, affinity).  We assessed the Silhouette Score, Inertia Score and other observations for each cluster, finally confirming that K-Means yielded best results.

We used the Elbow method to find that n_clusters = 4 was a balanced choice. For our final K-Means model, we used the default parameters except for init='random' and random_state=42.
With these models, we developed a class library to share with our client: yelpaffluence_nyc.py class YelpAffluence_NYC.  The Methods for YelpAffluence_NYC object include tools to fit the model to NYC data, query Yelp API for list of locations, plot results in a bar graph by price level, and plot results on an NYC map.
  

### Conclusion
- [Notebook 01 - Yelp Affluence Demo](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/01%20-%20Yelp%20Affluence%20Demo%20190118.ipynb)
- [Notebook 07 - Yelp Class Development](https://github.com/twludlow/portfolio-04-api-clustering/blob/master/07%20-%20Yelp%20Class%20Development.ipynb)

After optimizing our K-Means model, we deployed the K-Nearest Neighbors Classifier (n=15) to assign a label to our queried locations.  We found that the K-Means model sorted all NYC ZIP codes into 4 separate groups, which we have labeled "\$", "\$\$", "\$\$\$", and "\$\$\$\$" to match Yelp's existing pricing scale.

We noted that locations in each group shared distinct characteristics, and that our class designation presents a meaningful economic description for a particular location.

- **\$ - Red**
     - Limited economic activity across all price ranges (\$ - \$\$\$\$)
 
- **\$\$ - Orange**
     - Moderate economic activity in the \$ and \$\$ price levels
     - Limited activity in the \$\$\$ and \$\$\$\$ price categories
 
- **\$\$\$ - Green**
     - Highest activity in \$\$ price level
     - Increased activity in \$\$\$ and \$\$\$\$, but not highest
 
- **\$\$\$\$ - Blue**
     - Highest activity of all locations in \$\$\$ and \$\$\$\$ price establishments
     - High activity in \$\$ category
 
These trends are confirmed with inspection of the mean distribution of top-100 establishment price levels for each Yelp Affluence level.

##### Recommendations
- Feed other models with cluster results
     - This information can be useful as an economic rating variable in a predictive model

- Pay to use the Yelp Fusion VIP API
     - Will allow for commercial-scale queries

- Beware of ZIP query results
     - Yelp returned out of state results for some NYC-based zipcodes
     - e.g. 10015 returned results in Tucson, AZ
 
- Gathering and testing more data
     - Beyond the top 100, best match results

- Expanding class functionality
     - Enable collection of new training data
     - Automate model optimization

- Scaling the model
     - Train on other large metropolitan areas and check consistency of results


