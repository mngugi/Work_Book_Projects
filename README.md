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




**References and Resources:**
G.Wright(n.d). RFM analysis (recency, frequency, monetary). TechTarget, Data Management. https://www.techtarget.com/searchdatamanagement/definition/RFM-analysis
Reference link:
https://statso.io/rfm-analysis-case-study/

---


