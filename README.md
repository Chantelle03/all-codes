APP PY
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:10:28 2024

@author: boitumelo
"""

#print("Hello DARA 2024")

"""
print("DARA")
"""

"""
VARIABLES IN PYTHON

SN=2024 -integer
Age=25
kenya_code = 254
date=21
pi= 31.42 - float
Height = 5.8
Age = "twenty" - string
BG="CMB is interesting"

THIS IS WRONG VARIABLE FORMAT
2age=50
my age=50
my-age=50

-----------------------------------------------------------
3 IMPORTANT VARIABLE TYPES

1.INTEGER-WHOLE NUMBER: 50,-100
2.FLOAT - DECIMAL NUMBER:-0.5 1500.1
3.STRING - TEXT:"a","50"

"""
"""
HOW TO READ A CSV FILE IN PYTHON

import pandas
csv_file=pandas.read_csv("country_data.csv")
print(csv_file)

print(csv_file.describe())
x = 5
y = 3



print[x+y]


"""
import pandas

cat=pandas.read_csv("country_data.csv")
print(cat)

print(cat.describe())

print(cat.info())

"""
DATA SCIENCE -ETL (Extract,Transform,Load)

Methods:
    .describe()
    .info()
string-object
integer-int64
float-float64

panda will always convert data into Dataframe (table of data)

single coloumn of data -series
"""
print(cat.info())


STORING DATA
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:15:55 2024

@author: boitumelo
"""

"""
can only store one value
age=50

Data Storage Type:store multiple values:
    list
    dictionary
"""
B1=50
B2=30
B3=100
age1=50
age2=30
age3=100

age=[50,30,100]
print(age)

#to access a value inside

print(age[0])

print(min(age))
print(max(age))
print(len(age))
#print(mean(age))
age_avg=sum(age)/len(age)
print(age_avg)

my_list=[42,-2021,6.283,"tau",0.05]
print(my_list)
"""
my_list=[42,-2021,6.283,"tau",0.05]
"""

"""
List features
-add items to a list using append()
insert items at specific index
remove items
display a range of values

"""
my_list.append("cow")
my_list.append("skywalker")
my_list.insert(1,"Pombili")
my_list.remove("cow")
print(my_list[0:3])

"""
Dictionary
collection of key value pairs
each key is unique
similar hash maps
similar to lists, they dont have an index
you use keys to acsess them
it is unorderd
jkeys are like coloumn namesvalues are the
create a dictionary with{}

"""
my_dc={"frequency":3,"color":"blue"}
freq=[20,40,60]
color=["blue","green","yellow"]
my_dc= {"frequency":freq,"color":color}
person = {"name":"Pombili","country":"Ghana"}
print(person['name' ])

import pandas as pd

df= pd.DataFrame(my_dc)
print(df['frequency'].min())
print(df['frequency'].max())
print(df['frequency'].mean())

fitered_dta=df['frequency']>30
df=df[fitered_dta]


DAY 3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:42:23 2024

@author: boitumelo
"""
"""
import matplotlib.pyplot as plt

x=[0, 2, 4, 6, 8]
y=[3, 7, 9, 11, 13]

plt.plot(x,y,'o')
plt.xlabel('Chantelle')
plt.ylabel('King')
plt.title('I AM THE GIRL')
plt.show()

#HTML GRAPH

import plotly.express as px

x_line = [1, 2, 3, 4, 5]
y_line = [2, 4, 6, 8, 10]

fig = px.line(x=x_line, y=y_line, labels={'x': 'X-axis', 'y': 'Y-axis'}, title='Line Plot')
fig.write_html("plot.html")

# This is used to automatically open up a browser of your plot
import webbrowser
webbrowser.open("plot.html")

"""
"""
#radio freq
import numpy as np
import plotly.graph_objs as go
import webbrowser

# Generate synthetic frequency data (in MHz) and intensity (arbitrary units)
frequencies = np.linspace(0.1, 1000, 1000)  # 0.1 MHz to 1000 MHz
intensities = np.sin(frequencies / 50) ** 2 + np.random.normal(0, 0.05, frequencies.shape)

# Create a plotly figure
fig = go.Figure()

# Add the power spectrum as a line plot
fig.add_trace(go.Scatter(
    x=frequencies,
    y=intensities,
    mode='lines',
    name='Power Spectrum',
    line=dict(color='blue')
))

# Customize the layout
fig.update_layout(
    title="Radio Signal Power Spectrum",
    xaxis_title="Frequency (MHz)",
    yaxis_title="Intensity (arbitrary units)",
    # template="plotly_dark"
)

# Save the plot to an HTML file
fig.write_html("plot.html")

# Automatically open the plot in the default web browser
webbrowser.open("plot.html")
"""

import numpy as np

x=np.arange(0,10.1,0.1)
y1=x*x
y2=x**2*np.sin(x)
#x^2sin(x)
import matplotlib.pyplot as plt
plt.plot(x,y1,"r*")
plt.plot(x,y2,"g")
plt.show()

import numpy as np
x = np.arange(0,11)
y=x**2+3*x*np.random.rand(x.size)-1
#y=x^2+3x*R-1
p = np.polyfit(x,y,2)
xfit=np.arange(0,10.01,0.01)
yfit=np.polyval(p,xfit)
import matplotlib.pyplot as plt
plt.plot(x,y,"r*")
plt.plot(xfit,yfit,"k")
plt.show()

#ANOTHER PLOT 
import matplotlib.pyplot as plt

hours = [29, 9, 10, 38, 16, 26, 50, 10, 30, 33, 43, 2, 39, 15, 44, 29, 41, 15, 24, 50]
results = [65, 7, 8, 76, 23, 56, 100, 3, 74, 48, 73, 0, 62, 37, 74, 40, 90, 42, 58, 100]

model = np.polyfit(x, y, 1)
predict = np.poly1d(model)
hours_studied = 20
print(predict(hours_studied))


fig = plt.figure()
ax = plt.subplot(111)
ax.set_title("Student Hours vs Test Results")
ax.set_xlabel('Hours (H)')
ax.set_ylabel('Results (%)')
ax.grid()
plt.scatter(hours, results)
plt.show() 
    

ETL PRACTICE
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:55:49 2024

@author: boitumelo
"""

import pandas as pd

df = pd.read_csv("data_02/country_data_index.csv")

import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

#gives the coloum names as it had none
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None, names= column_names)


#This means the data within the text file uses semicolons to separate different values (columns).
df = pd.read_csv("data_02/Geospatial Data.txt",sep=";")


#reading an excel file
df = pd.read_excel("data_02/residentdoctors.xlsx")
#reading a json file
df = pd.read_json("data_02/student_data.json")


df = pd.read_csv("data_02/country_data_index.csv")
#To avoid the appearance of the "Unnamed: 0" column,use the index_col parameter to explicitly specify which column you want to use as the index.
df = pd.read_csv("data_02/country_data_index.csv",index_col=0)


# Column Headings
#no headings
df = pd.read_csv("data_02/patient_data.csv")
column_names = ["duration", "pulse", "max_pulse", "calories"]
df = pd.read_csv("data_02/patient_data.csv", header=None, names=column_names)


#Inconsistent Data Types & Names
#the data has caps,small, text with numbers etc
df = pd.read_excel("data_02/residentdoctors.xlsx")
# Step 1: Extract the lower end of the age range (digits only)
df['LOWER_AGE'] = df['AGEDIST'].str.extract('(\d+)-')
df['AGEDIST'].str.extract('(\d+)-')
# Step 2: Convert the new column to float withouty this they would be called a string
df['LOWER_AGE'] = df['LOWER_AGE'].astype(int)


#Working with Dates
df = pd.read_csv("data_02/time_series_data.csv")
#checking the data type of the date
# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

#OR
# Split the 'Date' column into separate columns for year, month, and day
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

#NANs and Wrong Formats
import pandas as pd

df = pd.read_csv('data_02/patient_data_dates.csv')

# Allows you to see all rows
pd.set_option('display.max_rows',None)

print(df)
"""

    The data set has an index column that is redundant
    The data set contains some empty cells or NaNs (“Date” in row 22, and “Calories” in row 18 and 28, “Maxpulse” in row 1).
    The data set contains wrong format (“Date” in row 26).
    The data set contains wrong data (“Duration” in row 7 and 13).
    The data set contains duplicates (row 11 and 12).
"""
#index column is redundant and we do not need it. We can remove it with drop 
df.drop(['Index'],inplace=True,axis=1)


#Replace Empty Values NANs- Using fillna
#inplace=True, it means that the changes made by the fillna operation will be applied directly to the original DataFrame (df in this case), and it will not return a new DataFrame. 
x = df["Calories"].mean()

df["Calories"].fillna(x, inplace = True) 



#Wrong Date Format – Convert with to_datetime()
df['Date'] = pd.to_datetime(df['Date'])

#One way to deal with empty values is simply removing the entire row using 
#df.dropna(inplace = True).

#to remove rows in that Date column you use 
df.dropna(subset=['Date'], inplace = True)


#Removing Empty Cells – Using dropna
#using the df.dropna() function. By default, the dropna() method returns a new DataFrame, and will not change the original
#If you want to change the original DataFrame, use the inplace = True argument. The dropna(inplace = True) will NOT return a new DataFrame, but it will remove all rows containing NULL values from the original DataFrame. You will also need to reset the index with df.reset_index(drop=True) as if you remove a row, the row numbers will not be consecutive:
df.dropna(inplace = True)
df = df.reset_index(drop=True)


#Wrong Data – Replace and Remove Rows

df.loc[7, 'Duration'] = 45
 #removed that row completely using df.drop(7, inplace = True).
 
 
 #Removing Duplicates – Using drop_duplicates()
df.drop_duplicates(inplace = True)





#SECTION 2
# aggregate data using groupby in pandas
 #append and merge datasets using different join types
    #filter and manipulate data to create new variables.
    
"""   
    #Aggregation
grouped = df.groupby('classs')

# Calculate mean, sum, and count for the squared values
mean_squared_values = grouped['sepal_length_sq'].mean()
sum_squared_values = grouped['sepal_length_sq'].sum()
count_squared_values = grouped['sepal_length_sq'].count()

# Display the results
print("Mean of Sepal Length Squared:")
print(mean_squared_values)

print("\nSum of Sepal Length Squared:")
print(sum_squared_values)

print("\nCount of Sepal Length Squared:")
print(count_squared_values)
"""
#Append & Merge
#this is when  they have similar coloumns
import pandas as pd

# Read the CSV files into dataframes
df1 = pd.read_csv("data_02/person_split1.csv")
df2 = pd.read_csv("data_02/person_split2.csv")

# Concatenate the dataframes
df = pd.concat([df1, df2], ignore_index=True)

#Append and Merge
#when they have different coloumns but there is a need to merge them
#Excel
df.to_excel("data_02/output/iris_data_cleaned.xlsx", index=False, sheet_name='Sheet1')

df1 = pd.read_csv('data_02/person_education.csv')
df2 = pd.read_csv('data_02/person_work.csv')

## inner join
df_merge = pd.merge(df1,df2,on='id')

#An outer join returns all the rows from both dataframes.
df_merge = pd.merge(df1, df2, on='id', how='outer')
"""
#Filtering
# Filter data for females (class == 'Iris-versicolor')
iris_versicolor = df[df['class'] == 'Iris-versicolor']

# Calculate the average iris_versicolor_sep_length
avg_iris_versicolor_sep_length = iris_versicolor['sepal_length'].mean()

#There is also a better way to label the "class" column since the word "Iris-" is redundant. We can remove it in the following way:
df['class'] = df['class'].str.replace('Iris-', '')

#If you have your own custom change you want to do to each value
# Apply the square to sepal length using a lambda function
df['sepal_length_sq'] = df['sepal_length'].apply(lambda x: x**2)
"""
#
    #Not filling in zeros - different to blank, a zero is actual data that was measured
    #Null Values - different to zero, null was not measured and thus should be ignored
    #Formatting to make data sheet pretty - highlighting and similar - add a new column instead with info
    #Comments in cells - place in separate column
    #Entering more than one piece of information in a cell - only one piece of information per cell
    #Using problematic field names - avoid spaces, numbers, and special characeters
   # Using special characters in data - avoid in your data


#FOR 
df.to_csv("data_02/output/iris_data_cleaned.csv")

#If you don't want the Pandas index column you can specify:

df.to_csv("data_02/output/iris_data_cleaned.csv", index=False)

#Excel

df.to_excel("data_02/output/iris_data_cleaned.xlsx", index=False, sheet_name='Sheet1')

#JSON
df.to_json("data_02/output/iris_data_cleaned.json", orient='records')


df.to_json("data_02/output/iris_data_cleaned.json", orient='records')

 

 

 


#column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

#df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None, names= column_names)




# all-codes
