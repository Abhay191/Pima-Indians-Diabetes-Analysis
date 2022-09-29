

# Name - Abhay Gupta
# registration no. - B20075
# Mobile - 9511334630

#import libraries
import pandas as pd
import matplotlib.pyplot as plt

#use pandas to read csv file
df1 = pd.read_csv('pima-indians-diabetes.csv',sep = ",")

# Question number 1

# make a function which takes dataframe as input and calculates mean,median,mode,minimum,maximun and standard deviation
# use inbuilt pandas dunction to calculate the above properties

def fun(df):
    print("Mean: ",df.mean())
    print("Median: ",df.median())
    print("Mode: ",df.mode())
    print("minimum: ",df.min())
    print("maximum: ",df.max())
    print("Standard deviation: ",df.std())
    
fun(df1)


# Question number 2

# Using matplotlib to get the scatter plots
def scatter1(df,s):
    # this function will plot histograms of attributed against the age attribute
    # s is the name of attribute and df is the column of dataframe which has to be plotted   
    plt.figure()
    plt.scatter(df1['Age'],df)
    plt.xlabel("Age")
    plt.ylabel(s)
    
scatter1(df1['pregs'],'pregs')
scatter1(df1['plas'],'plas')
scatter1(df1['pres'],'pres')
scatter1(df1['skin'],'skin')
scatter1(df1['test'],'test')
scatter1(df1['BMI'],'BMI')
scatter1(df1['pedi'],'pedi')


def scatter2(df,s):
    # this function will plot histograms of attributed against the BMI attribute
    # s is the name of attribute and df is the column of dataframe which has to be plotted 
    
    plt.figure()
    plt.scatter(df1['BMI'],df)
    plt.xlabel("BMI")
    plt.ylabel(s)

scatter2(df1['pregs'],'pregs')
scatter2(df1['plas'],'plas')
scatter2(df1['pres'],'pres')
scatter2(df1['skin'],'skin')
scatter2(df1['test'],'test')
scatter2(df1['pedi'],'pedi')
scatter2(df1['Age'],'Age')



# Question 3

# here we have to calculate the correlation coefficients
correl_df = df1.corr(method ='pearson')
print("Correlation coefficients between age and other properties: ")
print(correl_df['Age'])

print("Correlation coefficients between BMI and other properties: ")
print(correl_df['BMI'])


# Question 4

# use matplotlib to plot histograms
# make a function hist1 which takes column of dataframe and name of attribute as the input.
def hist1(df,s):
    plt.figure()
    plt.hist(df)
    plt.ylabel("Frequency")
    plt.xlabel(s)
    
hist1(df1['pregs'],'pregs') 
hist1(df1['skin'],'skin (in mm))')


# Question number 5

# use matplotlib to plot histograms
# make a function hist2 which takes column of dataframe and name of attribute as the input
def hist2(df,s):
    plt.figure()
    plt.hist(df,bins = [0,3,6,9,12,15,18])
    plt.xticks([0,3,6,9,12,15,18])
    plt.ylabel("Frequency")
    plt.xlabel(s)
    plt.grid(True)

# seperate the elements with class 0 and class 1
c1 = df1[df1['class'] == 1]
c0 = df1[df1['class'] == 0]

# plot the histograms by calling the function 
hist2(c0['pregs'],'pregs with class 0')
hist2(c1['pregs'],'pregs with class 1')


# Question number 6

# Use matplotlib to plot the boxplot
# Make a function which takes the column of dataframe as the input.

def box(df):
    plt.figure()
    plt.boxplot(df)
   
#plot the boxplot by calling the function 
box(df1['pregs'])
box(df1['plas'])
box(df1['pres'])
box(df1['skin'])
box(df1['test'])
box(df1['BMI'])
box(df1['pedi'])
box(df1['Age'])



