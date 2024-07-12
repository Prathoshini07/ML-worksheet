import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


data={
'Temperature(C)':[14.2,16.4,11.9,15.2,18.5,22.1,19.4,25.1,23.4,18.1,22.6,17.2],
 'Ice cream sales($)':[215,325,185,332,406,522,412,614,544,421,445,408]}
df=pd.DataFrame(data)
x=df[['Temperature(C)']]
y=df['Ice cream sales($)']
print(x)
print(y)

best_fit_line=LinearRegression().fit(x,y)
plt.scatter(x,y)
plt.plot(x,best_fit_line.predict(x))
plt.legend()
plt.show()

interpolate=21
interpolate_result=best_fit_line.predict([[interpolate]])
print(f"The value of interpolation of 21 : {interpolate_result[0]:.2f}")

extrapolate=29
extrapolate_result=best_fit_line.predict([[extrapolate]])
print(f"The value of extrapolation of 29 : {extrapolate_result[0]:.2f}")

