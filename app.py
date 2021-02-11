from flask import Flask, render_template , url_for , request , redirect
import numpy as np
import pickle


# loading trained model
model=pickle.load(open('model.pkl','rb'))

# initializing flask application
app=Flask(__name__)

# home page route(index.html)
@app.route('/')
def home():
    return render_template('index.html')

# when submitting the form 
@app.route('/form_action', methods=['POST','GET'])
def formAction():
    
    if request.method =='POST':
        values = request.form
        results = return_prediction(model,values) #stores the predicted flower name in results var
        return render_template('index.html', result = results) #return to the index page with results placeholder changed to the predicted name

    return render_template("index.html", result = "An error occured try again")

# func to predit the flower from the values got from form and returning the predicted flower name

def return_prediction(model,final):

    s_len=final["sLength"]
    s_wid=final["sWidth"]
    p_len=final["pLength"]
    p_wid=final["pWidth"]

    flower=[[s_len,s_wid,p_len,p_wid]]

    return model.predict(flower)

# runs the app
if __name__ == '__main__':
    app.run()    