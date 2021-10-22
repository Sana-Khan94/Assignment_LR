from flask import Flask, render_template, request,jsonify
from flask import Response
from flask_cors import cross_origin
import pickle
import pandas as pd

app = Flask(__name__)



@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            #CRIM	ZN	CHAS	NOX	RM	DIS	PTRATIO	B	LSTAT
            pclass=float(request.form['CRIM'])
            age = float(request.form['ZN'])
            sibsp = float(request.form['CHAS'])
            parch = float(request.form['NOX'])
            fare = float(request.form['RM'])
            class1 = float(request.form['DIS'])
            who = float(request.form['PTRATIO'])
            adult_male = float(request.form['B'])
            embark_town = float(request.form['LSTAT'])


            #  model = 'titanic_lr_mode.pickle'
            #stand= 'sandardScalar.sav'
            #trans= 'transform.sav'
           # loaded_stand = pickle.load(open(stand, 'rb'))
            #loaded_trans = pickle.load(open(trans, 'rb'))
            #  loaded_model = pickle.load(open(model, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file

           # t=transform([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,TAX,PTRATIO,LSTAT]])
            #s=loaded_stand.fit_transform(t)

            prediction=predict_lr([[pclass,age,sibsp,parch,fare,class1,who,adult_male,embark_town]])
            print('prediction is', prediction[0])
            # showing the prediction results in a UI
            return render_template('results.html',prediction=prediction[0])
        except Exception as e:
            print('The Exception message is: ',e)
            return e
    # return render_template('results.html')
    else:
        return render_template('index.html')


@app.route("/price_via_postman1",methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if request.json['data'] is not None:
            data = request.json['data']
            print('data is:     ', data)
            res = predict_lr(data)
            print('result is        ',res)
            return Response(res)
    except ValueError:
        return Response("Value not found")
    except Exception as e:
        print('exception is   ',e)
        return Response(e)


def predict_lr(dict_pred):
    if (request.method =='POST'):
        with open('linear_reg.pickle','rb') as f:
            model = pickle.load(f)

        data_df = pd.DataFrame(dict_pred, index=[1, ])
        predict = model.predict(data_df)
        return predict


if __name__=="__main__":
    host = '0.0.0.0'
    port = 8080
    app.run(debug=False)
