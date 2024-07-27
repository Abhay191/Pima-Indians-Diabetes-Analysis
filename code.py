# Name - Abhay Gupta
# registration no. - B20075
# Mobile - 9511334630

#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import statistics
#use pandas to read csv file
df1 = pd.read_csv('pima-indians-diabetes.csv',sep = ",")

# make a new dataframe excluding the class attribute

df2 = df1.copy(deep = True)


# Now we will identify outliers and replace them with median of the respective attributes

# Question 1
for col in df1.columns:
    q1 = df1[col].quantile(0.25)
    q3 = df1[col].quantile(0.75)
    iqr = q3-q1
    l_bound = q1-(1.5*iqr)
    u_bound = q3+(1.5*iqr)
    med = df1[col].median()
    for i in range(len(df2[col])):
        if(df2[col][i] < l_bound or df2[col][i] > u_bound):
            df2[col][i] = med
                             

# (a) part
df3 = df2.copy()
print('Question 1 :-')
print()
# printing minimum and maximum value of each atribute of dataframe before normalisation using the function df.min() and df.max()
print("Before min-max normalisation:")
print("Min:")
print(df3.min())
print()
print("Max:")
print(df3.max())
   
# defining a min_max function which will perform min-max normalisation on the dataframe .
# Min-max normalisation transforms data such that it falls within small specified range

def min_max(df,new_min,new_max):
    for col in df.columns:
        old_max = max(df[col])
        old_min = min(df[col])
        for i in range(len(df[col])):
            df[col][i] = ((df[col][i]-old_min)*(new_max-new_min)/(old_max-old_min))+new_min
    return df
min_max(df3,5,12)

print()
# printing minimum and maximum value of each atribute of dataframe after normalisation using the function df.min() and df.max()
print("After min-max normalisation:")
print("Min:")
print(df3.min())
print()
print("Max:")
print(df3.max())


# (b) part
df4 = df2.copy()

print()
# printing mean and standard deviation of each attribute before standardization by using df.mean() and df.std()
print('Before Standardization:')
print("Mean:")
print(df4.mean())
print()
print('Standard deviation:')
print(df4.std())
    
# defining a function 'standardize' which perform standardization process on a dataframe.
# Standardization is the process of rescalin data such that each attribute of new data has mean as 0 and variance as 1
def standardize(df):
    for col in df.columns:
        u = df[col].mean()
        s = df[col].std()
        df[col] = (df[col]-u)/s
    return df       

standardize(df4)

# printing mean and standard deviation of each attribute after standardization by using df.mean() and df.std()
print('After Standardization:')
print("Mean:")
print(df4.mean())
print()
print('Standard deviation:')
print(df4.std())


# Question 2

# (a) part
# generating a bi-variate Gaussian distribution sample.
mean = [0,0]
covr = [[13,-3],[-3,5]]
D = np.random.multivariate_normal(mean = [0,0],cov = covr,size = 1000)
D = pd.DataFrame(D,columns = ['x1','x2'])

#plotting a scatter plot between two attributes x1 and x2 using matplotlib
plt.figure()
plt.scatter(D['x1'],D['x2'])
plt.xlabel('x1') 
plt.ylabel('x2')
plt.show()

# (b) part
# finding eigen values and eigen vectors of the covariance matrix. eigen values are represented as w and eigen vectors as v.
cov = np.dot(np.transpose(D), D)/1000
w,v =  np.linalg.eig(cov)

# plotting the eigen directions on the scatter plot of the data.
plt.figure()
plt.scatter(D['x1'],D['x2'])
plt.xlabel('x1') 
plt.ylabel('x2')
plt.quiver(0,0,v[0][0],v[1][0],color = 'red',scale= 5)
plt.quiver(0,0,v[0][1],v[1][1],color = 'red',scale = 4)
plt.show()

# (c) part
A = np.dot(D,v)

for i in range(2):
    x = []
    y = []
    for k in A:
        # finding x coordinates of projected data
        x.append(k[i]*v[0][i])
        #finding y co-ordinated of projected data
        y.append(k[i]*v[1][i])
    # plotting the scatter plots superimposed on eigen vectors
    plt.figure()
    plt.scatter(D['x1'], D['x2'])
    plt.scatter(x, y)
    plt.quiver(0,0,v[0][0],v[1][0],color = 'red',scale= 5)
    plt.quiver(0,0,v[0][1],v[1][1],color = 'red',scale = 4)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# (d) part
# projecting data on eigen vectors using pca
pca = decomposition.PCA(n_components = 2)
data_proj = pca.fit_transform(D)

#reconstructing data using pca
D_re = pca.inverse_transform(data_proj)
D_re = pd.DataFrame(D_re,columns = ['x1_n','x2_n'])

#calculating euclidean distance between original and reconstructed data
euclid_dist = ((D_re['x1_n']-D['x1'])**2 +  (D_re['x2_n']-D['x2'])**2)**0.5

# calculating reconstruction error
reconst_error = (euclid_dist.sum())/1000
print('Reconstruction error = ',reconst_error)


# Question 3
# (a) part
df4 = df4.drop('class',axis = 1)

# reducing the 8 dimensional data on 2 dimensions
pca = decomposition.PCA(n_components = 2)
df3a = pca.fit_transform(df4)
df3a = pd.DataFrame(df3a, columns=['a1', 'a2']) 
print(df3a)

# plotting the scatter plot between 2 attributes of projected data
plt.figure()
plt.scatter(df3a['a1'], df3a['a2'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# calculating variance of each attribute
print('Variance of each attribute of reduced data is: ')
print(df3a.var())

# finding the covariance matrix and then the eigen values
covar_mat = np.dot(np.transpose(df4), df4)/len(df4)
e_value,vector = np.linalg.eig(covar_mat)
print('Eigen values are: ')
print(e_value[:2])

# (b) part
# plotting all eigen values in descending order
num = [1,2,3,4,5,6,7,8]
e_value = sorted(list(e_value), reverse = True) 
plt.figure()
plt.bar(list(range(len(e_value))),e_value)
plt.title('eigen values in descending order')
plt.xlabel("Eigenvalue")
plt.ylabel('magnitude')
plt.show()

# (c) part
# making an array to store the reconstruction errors for each l
errors = []

for i in range(1, 9):
    # projecting the data for each l
    pca = decomposition.PCA(n_components = i)
    data_projc = pca.fit_transform(df4)         
    data_projc = pd.DataFrame(data_projc) 
    # making a covariance matrix for each l     
    cov_mat = np.dot(np.transpose( data_projc),  data_projc)/len( data_projc)
    print(cov_mat)    
    # reconstructing the data
    data_rec = pca.inverse_transform(data_projc)
    data_rec = pd.DataFrame(data_rec)
    n = len(data_rec[0])
    e1 = [0]*n
    #calculating euclidean distance between original and reconstructed data
    for j in range(n):
        s = 0
        for k in range(i):
            s = s+(data_rec[k][j] - data_projc[k][j])**2
        e1.append(s**0.5)
    # finding the reconstruction error
    errors.append(statistics.mean(e1))
print(errors)

l = [1, 2, 3, 4, 5, 6, 7,8]
# plotting the line graph of reconstruction errors for different values of l
errors.reverse()
plt.figure()
plt.plot(l, errors)
plt.xlabel('Values of l')
plt.ylabel('Reconstruction Errors')
plt.show()

# (d) part
# finding 8 dimensional representation using pca then calculating it's covariance matrix
pca1 = decomposition.PCA(n_components = 8)
df3d = pca1.fit_transform(df4)
df3d = pd.DataFrame(df3d)
print(covar_mat)
covar_mat_new = np.dot(np.transpose(df3d), df3d)/len(df3d)
