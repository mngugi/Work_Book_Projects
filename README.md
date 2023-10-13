### Week 1 Project : Churn Prediction for Sprint.
**2023-09-28**

**Question 1**

Imagine you’re working with Sprint, one of the biggest Telecom companies in the USA. They’re really keen on figuring out how many customers might decide to leave them in the coming months. Luckily, they’ve got a bunch of past data about when customers have left before, as well as info about who these customers are, what they’ve bought, and other things like that. So, if you were in charge of predicting customer churn how would you go about using machine learning to make a good guess about which customers might leave? Like, what steps would you take to create a machine learning model that can predict if someone’s going to leave or not?

**Solution**

**1. Review Existing Customer Data:** 


The first step is to ’assemble’ all the existing data that pertains the customers who have left Churn Spirit and the currently existing Customers. Then categorize the data into two groups , Customers who have left and the Current Customers.Start by analyzing Customer usage patterns , observe Customer communication for example Customer complaints and feedback support given back to
the Customer and review Customer payment plans. In a nutshell these three attributes are the major reasons why a Customer would leave Sprint Telecom.


**2. Start Processing the Data:** Once step 1 above is thoroughly done the next step is to process data by cleaning the data ambiguity including data inconsistencies in order to have accurate data to work with. This processing involves data encoding that is ensuring all the data required is in numerical format and that a metric function can be used to process .

**3. Identify important elements that impact on Customers:** At this step it is important to identify which features impacts on the Customer to leave Spirit Telecom services. For example identifying relationships, correlations and models that are of impact to customers.


**4. Identify and Select a Model:** 
At this stage it important to identify which algorithm is suitable for prediction of the datasets obtained from step 1 and step 2 above. Some of the known Machine Learning algorithms are logistic regression, decision trees, random forests, gradient boosting , and neural networks.


**5. Model Training and Evaluate the Model:** 
Train the model according to the dataset obtained using the necessary algorithm. Evaluate the model performance using data tests cross validating on the accuracy of the results
produced.


**6. Interpretation, Visualization and Representation of Results:**

This step ensures that the results obtained are readable, repeatable and such can be interpretative, and visually represented to the consumer in this case Sprint Telecom.


**7. Project Execution and Deployment:** 
The algorithm is ready for deployment and needs to be implemented on real time customer data for continuous data retraining and continuous improvement as well tracking on the
results.


**8. Maintenance and Feedback:** Continuous re-evaluation of the process against feedback from the customers is import for customer retention and to identify solutions as to why they customers want to leave.

### Implentation 
``` python
a = '''
\033[1mWeek1 Write up : Lux Academy Data Science Bootcamp\033[0m
\033[1mProject Name : Churn Prediction for Sprint Telecom\033[0m
\033[1mAuthor: Peter Mwangi Ngugi\033[0m

\033[1mProblem Defination:\033[0m

\033[1mProblem Description:\033[0m
Sprint Telecom one of the biggest Telecom companies in the USA are keen on figuring out how many customers 
might decide to leave them in the coming months. 

Luckily, they’ve got a bunch of past data about when customers have left before, as well as info about 
who these customers are, what they’ve bought, and other things like that. 

\033[1mObjectives:\033[0m

So, if you were in charge of predicting customer churn how would you go about using machine learning to make a good guess about which 
customers might leave? Like, what steps would you take to create a machine learning model that can predict 
if someone’s going to leave or not?
'''

print(a)

# Required python libraries for this task are:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# using pandas to display the  entire Data Frame
# the data set used in this task is obtained and downloaded from: 
# https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics?select=telecom_zipcode_population.csv 

b= '''
The first step is to have a review of the entire data set, so that a decision is made 
for considering which data will be of importance to our objectives. 
'''
print(b)
dt = pd.read_csv("/home/x38fed/Documents/GitHub/DataScienceBootCampLuxAcademy/Work_Book_Projects/telecom_customer_churn.csv")
 
# to display the entire DataFrame in a single cell
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Confirmation of the total number of Customers by calling the function dt.head(7043)
#print('The total number of Customers is:') dt.head(7043)
# Calculate and print the total number of customers
total_customers = dt.shape[0]
print('The total number of Customers is:', total_customers)
dt.head()

c = '''
It is observed that the important data that requires categaorization in relation to custer churn are :
['Customer Status', 'Churn Category', 'Churn Reason]
'''
print(c)

d = '''

\033[1mData Cleanup : At this stage a data cleanup is performed by:\033[0m 
1. Identifying the rows with null (NaN) values for the required columns.
2. Filtering the null values from the rows.
3. Creating a new Data frame that is of vital importance to this analysis.
'''

# At this point a variaable categoryCols is assigned 'Customer Status', 'Churn Category', 'Churn Reason columns
# from the main dataset table.

categoryCols = ['Customer Status', 'Churn Category', 'Churn Reason']
                
# important to create a new data frame for the selected columns
           
selectedColumns = dt[categoryCols]

# important to show all the rows to avoid missing important data
pd.set_option('display.max_rows', None)

# Create a boolean mask to identify rows with NaN values
nan_rows = selectedColumns.isnull().any(axis=1)

# Filter rows with NaN values and create a new DataFrame
filteredNaN = selectedColumns[~nan_rows]

# Display the filtered DataFrame
print(d) # this code is the output for the required tasks.
# this code prints the filtered NaNs
print(filteredNaN)

f = '''
\033[1m Data Visualization\033[0m

In order to further understand the data set and further get important relationship 
We share perform data exloration using visualization of the graph types:
- Histograms, pairplot and a piechart.
- The following data frame visual charts are created in the follwoing arrangement:
    \u2022 Churn Category Counts
    \u2022 Count of Churn category 
    \u2022 Churn Category Distribution pie chart
- Notable observation trend and of importance to Churn analysis is the \033[1mDistribution
of Number of Referrals Histogram graph\033[0m which demonstastates a reducing trend in the 
number of customer referrals.

'''
print(f)
# Create a DataFrame from the counts data
data = {'Churn Category': ['Competitor', 'Dissatisfaction', 'Attitude', 'Price', 'Other'],
        'Count': [841, 321, 314, 211, 182]}

df_counts = pd.DataFrame(data)

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(df_counts['Churn Category'], df_counts['Count'], color='skyblue')
plt.xlabel('Churn Category')
plt.ylabel('Count')
plt.title('Churn Category Counts')
plt.xticks(rotation=45)
plt.show()

categorical_column = 'Churn Category'
plt.figure(figsize=(8, 5))
sns.countplot(data=dt, x=categorical_column, palette='viridis')
plt.title(f'Count of {categorical_column}')
plt.xticks(rotation=45)
plt.show()

#create a pie chat
plt.figure(figsize=(6, 6))
plt.pie(df_counts['Count'], labels=df_counts['Churn Category'], autopct='%1.1f%%', colors=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'lightpink'])
plt.title('Churn Category Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

g = '''
\033[1mFrom the pie chart distribution we conclude that:\033[0m
1. Competitor reason which is 45%,is the largest reason why customers churn. This means competitors are offering better 
deals and services that are attracting customers from Sprint Telecom. It is recommended that Sprint improves of factors 
that should retain customers by under studying their competitors.
2. Dissatisfaction reason makes 17.2% of customers who churn. It is important to identfy and improve customer service 
that leads to dissatisfaction.
3. Attitue makes 16.8% this something to do with poor or negative feedback and customer support. Addressing cutomer needs
in a positive manner is required to improve relationships with customers.
4. Price is 11.3% this shows that the price is high and not competitive compared to that one of the competitor.It is import
to review the pricing in reference to those one of the competitor.
5. Other reasons constitute to 9.7% of why customers churn. The other reasons need to identofied and resolved.

'''
print(g)

import seaborn as sns
# Data Visualization
# Visualize the distribution of numeric columns
numeric_columns = dt.select_dtypes(include=['float64', 'int64'])
num_cols_to_visualize = min(len(numeric_columns.columns), 6)  # Limit to 6 columns for subplots

plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_columns.columns[:num_cols_to_visualize], 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=numeric_columns, x=col, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

h = '''
\033[1mRelationships Visualization\033[0m
Correlation Matrix table is important for the visualization of relationships 
between numeric variables.
At this point we need to look it has an impact on why customers churn.
\033[1mCorrection Spectrum: \033[m
|---------------------------------------------|
|Accordint to the Correlation Spectrum:       |
|> Perfect positive is a range of 1.00 - 0.75 |
|> Strong positive is a range of 0.50 - 0.25  |
|> Weak positive is  a range of 0.25 - 0.00   |
-----------------------------------------------
|> Perfect negative is a range of 1.00 - 0.75 |
|> Strong negative is a range of 0.50 - 0.25  |
|> Weak negative is  a range of 0.25 - 0.00   |
|---------------------------------------------|

\033[1mFrom this data correlation and in relation to customer churn it can be concluded that:
\u2022 That the relationship between the number of referrals and tenure of maonths is 1.00 to 0.33
which is moderately strong, and can be a exploited to strengthen the relationship between Sprint Telecom 
and her Customers.
\u2022 The  correlation between Monthly Charge, Number of refferels , total charge and tenure in months
is 0.25 to 0.83 and 0.24. This demonstrates a normal pattern for normal revenue correction. 
\u2023 Finally according to correlation matrix relationship cost of services and subscriptions provided 
for by Sprint Telecom are the major factor as to why Customers churn. It is this area where significanr 
attention and changes is required.


'''
print(h)


# Data Visualization
# Visualize the distribution of numeric columns
dt = pd.read_csv("/home/x38fed/Documents/GitHub/DataScienceBootCampLuxAcademy/Work_Book_Projects/telecom_customer_churn.csv")
numeric_columns = dt.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_columns.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix (Numeric Columns)")
plt.show()
                
# Code not complete 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

dt = pd.read_csv("/home/x38fed/Documents/GitHub/DataScienceBootCampLuxAcademy/Work_Book_Projects/telecom_customer_churn.csv")
# Specify the list of feature columns from your dataset
feature_columns = ['Churn Reason']  # Add more feature columns as needed

# Assuming 'X' contains your feature columns and 'y' contains your target column ('Churn Category')
X = dt[feature_columns]
y = dt['Churn Category']

# Drop rows with missing values
X = X.dropna()
y = y[X.index]  # Match the target labels with the filtered features

# Apply one-hot encoding to the 'Churn Reason' column
X = pd.get_dummies(X, columns=['Churn Reason'], drop_first=True)

# Split the data into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_rep)

h = ''' \033[1mReferences:\033[0m
 Peter Mwangi Ngugi Github Python_programming repository. https://github.com/mngugi/Python_Programming-/wiki
 Harun Mbaabu Github Data-Science-For-EveryOne . https://github.com/HarunMbaabu/Data-Science-For-EveryOne
 Python Code generation for predicting model accuracy is borrowed fron chatgpt. (Not completed)
 Data Set used is from Telecom Customer Churn Prediction
 https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics?select=telecom_zipcode_population.csv
  
 '''
print(h)                
                
```

---
### Week 2 Task RFM Analysis
**Problem Statement**


RFM analysis is a powerful technique used by companies to better understand customer behavior and optimize engagement strategies. It revolves around three key dimensions: 
recency, frequency, and monetary value. These dimensions capture essential aspects of customer transactions, providing valuable information for segmentation and personalized marketing campaigns.

The given dataset is provided by an e-commerce platform containing customer transaction data including customer ID, purchase date, transaction amount, product information, ID 
command and location. The platform aims to leverage RFM (recency, frequency, monetary value) analysis to segment customers and optimize customer engagement strategies.

Your task is to perform RFM analysis and develop customer segments based on their RFM scores.The analysis should provide insights into customer behaviour and identification of high-value customers,at-risk customers, and potential opportunities for personalized marketing 
campaigns.



**Understanding  RFM Analysis System.**

According to G.Wright(n.d). 

“RFM anlysis is a marketing technique used to quantitatively rank and group customers based on the recency, frequenc and monetary total of their recent transactions to identify the best customers and perform targeted marketing campaigns.”

RFM The system assigns each customer numerical scores based on these factors to provide an objective analysis. RFM analysis is based on the marketing adage that "80% of your business comes from 20% of your customers."_

**RFM Analysis**

“RFM analysis scores customers on each of the three main factors. Generally, a score from 1 to 5 is given, with 5 being the highest.”

**Definitions:**

    • Recency is the most recent time a customer purchased an item and it is measured in days, weeks, hours and years.

    • Frequency is how often a customer purchase an item.

    • Monetary this is how much a customer spends in a given period of time. 
      
    • Segmentation of customers in RFM analysis.  This is identifying clusters of customers with similar attributes.

**Customer Types**

    1. Whales : These are customers that tick all the three attributes with a high score of (5,5,5).
    2. New Customers: Customers with a high recency but a low frequency and obviously low monetary value. (5. 1 , x)
    3. Lapsed Customers: Customers with low recency but high value (1,X,5) were once valuable customers but have since stopped.

 
**Further Examples of Customer Segmentation**

**Loyal High Value Customer:** Most valuable Customers that should always get the highest priority customer engagements. 
**At-risk Customers:** They are customers with very low frequency, recency and monetary and they are often at risk for churning.
**New High- Value Customers:** These are customers with high monetary value but low on frequency and recency.
**Lapsed Customers:** Customers who ahve not purchased in a long time.


### Implementation and Solution 

```python
b = '''
\033[1mReading dataset and arranging the data in their respective columns.
\u2023 This is achieved by using the set_option function set into None and False.
\u2023 This is part of data preparation.\033[0m

'''
print(b)

import pandas as pd

# read the CSV file
data = pd.read_csv("rfm_data.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


# Check the rows of DataFrame
print(data.head(1000))

import pandas as pd

# Read the CSV file
data = pd.read_csv("rfm_data.csv")
c = '''

\033[1m Calculating Recency: This is achieved by by calculating the time since the last transaction.\033[1m

\033[1m Algorithm Calculate Recency:\033[1m

\u2022 Create a function that takes argument data, reference  date .
\u2022 The last date in the data set is 2023-06-10. 
\u2022 Define the latestDate by using pandas function to_datatime. latestDate = pd.to_datetime('2023-06-10')
\u2022 Transform PurchaseDate to a PurchaseDate column using pd.to_datetime(data['PurchaseDate'])
\u2022 Calculate Recency by subtracting Purchase date from latest date
\u2022 print the updated data set.
'''

print(c)

# Calculate Recency
def cal_Recency(data, ref_date='2023-06-10'):
    latestDate = pd.to_datetime('2023-06-10')
    data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'])
    data['Recency'] = (latestDate - data['PurchaseDate']).dt.days 
    return data

data = cal_Recency(data)

d = ''' 
\033[1m Calculate frequency: this is done by tabulating the number of transactions per customer.\033[1m

\033[1m Algorithm for Calculating frequency:\033[1m 

\u2022 Define a function named cal_Frequency that takes data argument.
\u2022 Calculate the frequency by regrouping data based on CustomerID column and counting 
  the number of orderID values for each group.
\u2022 Merge frquency data with the original data.
\u2022 Return data if everything is correct. 

'''
# Calculate Frequency
def cal_Frequency(data):
    frequencyData = data.groupby('CustomerID')['OrderID'].count().reset_index()
    frequencyData.rename(columns={'OrderID': 'Frequency'}, inplace=True)
    data = data.merge(frequencyData, on='CustomerID', how='left')
    return data

data = cal_Frequency(data)
e = '''
\033[1m Calculate Monetary: this daone by calculating the total monetary value of transactions
per customers. \033[1m

\033[1m Algorithm for Calculating Monetary Value:

\u2022 Define the function named cal_Monetary that takes data as the argument.
\u2022 Calculate the TransactionAmount by adding the sum for each group.
\u2022 Merge monetary data with the original data.
\u2022 Return data if everything is correct.

'''
# Calculate Monetary
def cal_Monetary(data):
    monetaryData = data.groupby('CustomerID')['TransactionAmount'].sum().reset_index()
    monetaryData.rename(columns={'TransactionAmount': 'Monetary'}, inplace=True)
    data = data.merge(monetaryData, on='CustomerID', how='left')
    return data

data = cal_Monetary(data)
print(d)
print(e)

f = ''' 
\033[1m \u2023RFM analysis scores for customers on each of the three main factors.Generally, a score from 1 to 5 is given, with 5 being the highest.\033[1m

\033[1m \u2027 At this step we shall use the following steps:
\u2027 Define the scoring data that is a range of 1 - 5, with 1 the lowest and 5 the highest.
\u2027 The third step is to calculate the RFM scores.
\u2027 The final part of tabaulating data is to calculate the Segment score
'''

# RFM Analysis Scores
recencyScores = [1, 2, 3, 4, 5]
frequencyScores = [1, 2, 3, 4, 5]
monetaryScores = [1, 2, 3, 4, 5]

data['RecencyScore'] = pd.qcut(data['Recency'], q=5, labels=False, duplicates='drop') + 1
data['FrequencyScore'] = pd.qcut(data['Frequency'], q=5, labels=False, duplicates='drop') + 1
data['MonetaryScore'] = pd.qcut(data['Monetary'], q=5, labels=False, duplicates='drop') + 1

# Calculate the RFM scores
def calculate_rfm_scores(data):
    data['RFM_Score'] = data['RecencyScore'] + data['FrequencyScore'] + data['MonetaryScore']
    segment_labels = ['Low-Value', 'Mid-Value', 'High-Value']
    data['Value Segment'] = pd.qcut(data['RFM_Score'], q=3, labels=segment_labels)
    return data

data = calculate_rfm_scores(data)

# Display the DataFrame with RFM scores
pd.set_option('display.expand_frame_repr', None)
print(data)

# Convert RFM scores to numeric type
data['RecencyScore'] = data['RecencyScore'].astype(int)
data['FrequencyScore'] = data['FrequencyScore'].astype(int)
data['MonetaryScore'] = data['MonetaryScore'].astype(int)


import plotly.express as px

def plot_rfm_segment_distribution(data):
    # Calculate the counts of each RFM segment
    segment_counts = data['Value Segment'].value_counts().reset_index()
    segment_counts.columns = ['Value Segment', 'Count']

    # Define colors for the segments
    pastel_colors = px.colors.qualitative.Pastel

    # Create the bar chart
    fig_segment_dist = px.bar(segment_counts, x='Value Segment', y='Count', 
                              color='Value Segment', color_discrete_sequence=pastel_colors,
                              title='RFM Value Segment Distribution')

    # Update the layout
    fig_segment_dist.update_layout(xaxis_title='RFM Value Segment',
                                  yaxis_title='Count',
                                  showlegend=False)

    # Show or return the chart, depending on your needs
    return fig_segment_dist

# Example usage:
fig = plot_rfm_segment_distribution(data)
fig.show()  # To display the chart


import plotly.express as px
import plotly.graph_objects as go

# Set the Plotly template
import plotly.io as pio
pio.templates.default = "plotly_white"

# 1. Calculate RFM Customer Segments
data['RFM Customer Segments'] = ''

# Assign RFM segments based on the RFM score
data.loc[data['RFM_Score'] >= 9, 'RFM Customer Segments'] = 'Champions'
data.loc[(data['RFM_Score'] >= 6) & (data['RFM_Score'] < 9), 'RFM Customer Segments'] = 'Potential Loyalists'
data.loc[(data['RFM_Score'] >= 5) & (data['RFM_Score'] < 6), 'RFM Customer Segments'] = 'At Risk Customers'
data.loc[(data['RFM_Score'] >= 4) & (data['RFM_Score'] < 5), 'RFM Customer Segments'] = "Can't Lose"
data.loc[(data['RFM_Score'] >= 3) & (data['RFM_Score'] < 4), 'RFM Customer Segments'] = "Lost"

# Print the updated data with RFM segments
pd.set_option('display.expand_frame_repr', None)
print(data[['CustomerID', 'RFM Customer Segments']])

# 2. RFM Segment Distribution Treemap
segment_product_counts = data.groupby(['Value Segment', 'RFM Customer Segments']).size().reset_index(name='Count')
segment_product_counts = segment_product_counts.sort_values('Count', ascending=False)

fig_treemap_segment_product = px.treemap(segment_product_counts, 
                                         path=['Value Segment', 'RFM Customer Segments'], 
                                         values='Count',
                                         color='Value Segment', color_discrete_sequence=px.colors.qualitative.Pastel,
                                         title='RFM Customer Segments by Value')

# 3. Distribution of RFM Values within Champions Segment (Box Plots)
# Filter the data to include only the customers in the Champions segment
champions_segment = data[data['RFM Customer Segments'] == 'Champions']

fig_box_champions = go.Figure()
fig_box_champions.add_trace(go.Box(y=champions_segment['RecencyScore'], name='Recency'))
fig_box_champions.add_trace(go.Box(y=champions_segment['FrequencyScore'], name='Frequency'))
fig_box_champions.add_trace(go.Box(y=champions_segment['MonetaryScore'], name='Monetary'))

fig_box_champions.update_layout(title='Distribution of RFM Values within Champions Segment',
                                yaxis_title='RFM Value',
                                showlegend=True)

# 4. Comparison of RFM Segments (Bar Chart)
import plotly.colors
pastel_colors = plotly.colors.qualitative.Pastel

segment_counts = data['RFM Customer Segments'].value_counts()

fig_segment_counts = go.Figure(data=[go.Bar(x=segment_counts.index, y=segment_counts.values,
                            marker=dict(color=pastel_colors))])

# Set the color of the Champions segment as a different color
champions_color = 'rgb(158, 202, 225)'
fig_segment_counts.update_traces(marker_color=[champions_color if segment == 'Champions' else pastel_colors[i]
                                for i, segment in enumerate(segment_counts.index)],
                  marker_line_color='rgb(8, 48, 107)',
                  marker_line_width=1.5, opacity=0.6)

fig_segment_counts.update_layout(title='Comparison of RFM Segments',
                  xaxis_title='RFM Segments',
                  yaxis_title='Number of Customers',
                  showlegend=False)

# 5. Comparison of RFM Segments based on Recency, Frequency, and Monetary Scores (Grouped Bar Chart)
segment_scores = data.groupby('RFM Customer Segments')[['RecencyScore', 'FrequencyScore', 'MonetaryScore']].mean().reset_index()


fig_segment_scores = go.Figure()

# Add bars for Recency score
fig_segment_scores.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['RecencyScore'],
    name='Recency Score',
    marker_color='rgb(158,202,225)'
))

# Add bars for Frequency score
fig_segment_scores.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['FrequencyScore'],
    name='Frequency Score',
    marker_color='rgb(94,158,217)'
))

# Add bars for Monetary score
fig_segment_scores.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['MonetaryScore'],
    name='Monetary Score',
    marker_color='rgb(32,102,148)'
))

fig_segment_scores.update_layout(
    title='Comparison of RFM Segments based on Recency, Frequency, and Monetary Scores',
    xaxis_title='RFM Segments',
    yaxis_title='Score',
    barmode='group',
    showlegend=True)

# Display the figures
fig_treemap_segment_product.show()
fig_box_champions.show()
fig_segment_counts.show()
fig_segment_scores.show()



```


**References and Resources:**


G.Wright(n.d). RFM analysis (recency, frequency, monetary). TechTarget, Data Management. https://www.techtarget.com/searchdatamanagement/definition/RFM-analysis


Reference link:

https://statso.io/rfm-analysis-case-study/

```Html
Reference Notes:
Cell 20 and 21 Code i have used Aman Kharwal and Chat GPT code structure for data visualization . 
Matplot library could not generate the desired results and i need time on it. 
```
---


