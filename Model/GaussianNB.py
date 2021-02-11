import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle
import warnings

warnings.filterwarnings("ignore")

iris=pd.read_csv("iris.csv")
X = iris['Sepal.Length'].values.reshape(-1,1)
Y = iris['Sepal.Width'].values.reshape(-1,1)
train, test = train_test_split(iris, test_size = 0.25)

train_X = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length','Petal.Width']]
train_y = train.Species
test_X = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length','Petal.Width']]
test_y = test.Species

model = GaussianNB()
model.fit(train_X,train_y) 


inputt=[float(x) for x in "5.9 3.0 5.1 1.8".split(' ')]
final=[np.array(inputt)]

b = model.predict_proba(final)

print(model.predict(final))
print(b)

pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
