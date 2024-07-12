import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r"C:\Users\prath\Downloads\archive\advertising.csv")
print(df.columns)

tv_col = 'TV'
sales_col = 'Sales'

correlation = df[tv_col].corr(df[sales_col])
print(f"Correlation coefficient between sales and TV budget: {correlation:.2f}")

X = df[[tv_col]]
y = df[sales_col]
reg = LinearRegression().fit(X, y)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, reg.predict(X), color='red', label='Line of best fit')
plt.xlabel('TV Budget ($)')
plt.ylabel('Sales ($)')
plt.title('Sales vs. TV Budget')
plt.legend()
plt.show()

print(f"For every $1 increase in TV budget, sales are expected to increase by ${m:.2f}")

confidence_level = 0.95
t_value = stats.t.ppf((1 + confidence_level) / 2, len(X) - 2)
margin_error = t_value * np.std(y) / np.sqrt(len(X))

print(f"Confidence interval for the slope: ({m - margin_error:.2f}, {m + margin_error:.2f})")

if m - margin_error > 0 and m + margin_error > 0:
    print("Increasing sales through advertising has a positive impact on sales.")
elif m - margin_error < 0 and m + margin_error < 0:
    print("Increasing sales through advertising has a negative impact on sales.")
else:
    print("We are not confident that increasing sales through advertising has a significant impact on sales.")