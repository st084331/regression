from math import sqrt, exp, fabs
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

x = []
y = []

with open('population_by_country_2020.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if (row)['Fert'] != 'N.A.':
            x.append(float(row['Density (P/Km²)']))
            y.append(float(row['Fert']))

X = np.array(x).reshape(-1,1)
Y = np.array(y)
model1 = LinearRegression().fit(X, Y)
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
model2 = LinearRegression().fit(x_poly,Y)
index_0 = []
i = 0
y_0 = y
x_0 = x
for y0 in y:
    if y0 == 0:
        x_0.pop(i)
        y_0.pop(i)
        i-=1
    i+=1
expo = np.polyfit(x_0, np.log(y_0), 1, w=np.sqrt(y_0))
y_pred3 = []
for x1 in x:
    y_pred3.append(exp(expo[1]) * exp(expo[0] * x1))
y_pred1 = model1.predict(X).tolist()
y_pred2 = model2.predict(x_poly).tolist()

dists1 = []
dists2 = []
dists3 = []
for j in range(len(x)):
    dists1.append(fabs(y_pred1[j] - y[j]))
    dists2.append(fabs(y_pred2[j] - y[j]))
    dists3.append(fabs(y_pred3[j] - y[j]))
dists = [np.mean(dists1),np.mean(dists2),np.mean(dists3)]
min_dist = float('inf')
i = 0
indexs = []
for dist in dists:
    if dist < min_dist:
        min_dist = dist
for dist in dists:
    if dist > min_dist - 0.01 and dist < min_dist + 0.01:
        indexs.append(i)
    i+=1
print(indexs)
if indexs.count(0) != 0:
    print('Линейное приближение достаточно хорошо')
if indexs.count(1) != 0:
    print('Полиномиальное приближение достаточно хорошо')
if indexs.count(2) != 0:
    print('Экспотенциальное приближение достаточно хорошо')

size = 4
trans = 1

for i in range(len(x)):
    scatter1 = plt.scatter(x[i], y[i], c='blue', s = size, alpha = trans)
    scatter1 = plt.scatter(x[i], y_pred1[i], c = 'red', s = size, alpha = trans)
    scatter1 = plt.scatter(x[i], y_pred2[i], c = 'green', s = size, alpha = trans)
    scatter1 = plt.scatter(x[i], y_pred3[i], c = 'yellow', s = size, alpha = trans)
plt.show()