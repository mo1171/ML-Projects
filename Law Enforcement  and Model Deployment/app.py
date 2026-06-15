import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,BooleanField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
from CustomTranformers.Custom_Transf import Preprocces \
, FullImputer , preproccesBeforePipe ,getting_lat_long
from CustomTranformers.Checks import * 

########################################
# Begin database stuff

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    probability = FloatField()
    predicted_outcome = BooleanField()
    outcome = BooleanField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)
# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open(os.path.join('pickles', 'columns.json')) as fh:
    columns = json.load(fh)


with open(os.path.join('pickles', 'GBC_pipe.pickle'), 'rb') as fh:
    pipeline = joblib.load(fh)


with open(os.path.join('pickles', 'dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)
    
# with open(os.path.join('pickles', 'boundriesOfNumeric.json')) as fh:
#     boundriesOfNumeric = json.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/should_search', methods=['POST'])
def should_search():
    req = request.get_json()
  
    request_ok, error = check_request(req)
    if not request_ok:
        response = {'observation_id': req.get('observation_id') ,'error': error}
        return jsonify(response)
    _id = req['observation_id']
    observation = req.copy()
    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'observation_id':_id  ,'error': error}
        return jsonify(response)
    obs = pd.DataFrame([observation], columns=columns)
    try :
        obs = obs.astype(dtypes)
    except ValueError as e  :
        error_msg = 'observation has invalid data type which is  ' + f'{e}' 
        return jsonify({'error': error_msg})
    observation = preproccesBeforePipe(obs.iloc[[0]]).squeeze().to_dict()
    categories_ok, error,legislation_na = check_categorical_values(observation)
    if not categories_ok:
        response = {'observation_id':_id  ,'error': error}
        return jsonify(response)
    if legislation_na : 
        error = 'Ligislation was null but we deal with it, be carful when entering '
        response = {'warning': error}
    date_ok, error = check_date(observation)
    if not date_ok:
        response = {'observation_id':_id  ,'error': error}
        return jsonify(response)

    Poperaton_ok, error = check_Poperation(observation)
    if not Poperaton_ok:
        response = {'observation_id':_id  ,'error': error}
        return jsonify(response)
    else : 
        if error :
            response = {'warning': error}
    long_ok, error = check_long(observation)
    if not long_ok:
        response = {'observation_id':_id  ,'error': error}
        return jsonify(response)
    else : 
        if error :
            response = {'warning': error}
    lat_ok, error = check_lat(observation)
    if not lat_ok:
        response = {'observation_id':_id  ,'error': error}
        return jsonify(response)
    else : 
        if error :
            response = {'warning': error}
    
    threshold = .1  
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = 1 if proba > threshold else 0 
    response = {'outcome': bool(prediction),'proba' : proba}
    p = Prediction(
        observation_id=_id,
        observation=request.data,
        probability = proba ,
        predicted_outcome=bool(prediction)
        
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)

    
@app.route('/search_result', methods=['POST'])
def search_result():
    obs = request.get_json()
    request_ok, error = check_request(obs)
    if not request_ok:
        response = {'observation_id': obs.get('observation_id') ,'error': error}
        return jsonify(response)
    _id = obs['observation_id']
    columns_ok, error = check_valid_update(obs)
    if not columns_ok:
        response = {'observation_id':_id  ,'error': error}
        return jsonify(response)
    
    outcome_ok, error = check_True_outcome(obs)
    if not outcome_ok:
        response = {'observation_id':_id  ,'error': error}
        return jsonify(response)
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.outcome = obs['outcome']
        p.save()
        return jsonify({key : val for key ,val in  model_to_dict(p).items() if key not in ['observation','id']} )
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})

@app.route('/list-db-contents', methods=['POST'])
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)

