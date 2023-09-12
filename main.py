from sklearn import model_selection
import matplotlib.pyplot as pit
import numpy as np
from sklearn import linear_model
#y=mx+c
#F=1.8*C+32

x=list(range(0,10))  #C
y=[1.8*F + 32 for F in x]  #F
print(f'X: {x}')
print(f'Y: {y}')

pit.plot(x,y,'-*r')
pit.show()

x=np.array(x).reshape(-1,1)
y=np.array(y).reshape(-1,1)
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=0)

model=linear_model.LinearRegression()
model.fit(x_train,y_train)
accuracy=model.score(x_test,y_test)
print(accuracy*100)

