#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x=np.array([[1,2],
        	[1,3],
        	[1,5],
        	[1,7],
        	[1,9]])
y=np.array([[4],[5],[7],[10],[15]])
Y=np.array([4,5,7,10,15])
X=np.array([2,3,5,7,9])
XtransposeX=np.dot(np.transpose(x),x)
Xtransposey=np.dot(np.transpose(x),y)
b=np.dot(np.linalg.inv(XtransposeX),Xtransposey)
print(b)
print('THE LINEAR REGRESSION EQUATION IS y=',b[0],'+',b[1],'x')

r=np.corrcoef(X,Y)
print(r[0,1],' is the correlation coefficient')

#To find the linear regression using formula
sum_x=sum(X)
sum_y=sum(Y)
sum_xy=sum(X*Y)
sum_xsq=sum(X**2)
b_formula=((5*sum_xy)-(sum_x*sum_y))/((5*sum_xsq)-(sum_x**2))
a_formula=(sum_y-(b_formula*sum_x))/5
print(b_formula)
print(a_formula)
print('THE LINEAR REGRESSION EQUATION IS y=',a_formula,'+',b_formula,'x')
error=0

#Calculation of Mean square error
for i in range(5):
	error+=(Y[i]-(a_formula+b_formula*X[i]))**2
print('MSE',error)
