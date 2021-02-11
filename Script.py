import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

model=pickle.load(open('model.pkl','rb'))

def return_prediction(model,final):
    return model.predict(final)


s_len = input("enter sepal length")
s_wid = input("enter sepal width")
p_len = input("enter petal length")
p_wid = input("enter petal width")

inputt=[s_len,s_wid,p_len,p_wid]
final=[np.array(inputt)]
print(return_prediction(model,final))   