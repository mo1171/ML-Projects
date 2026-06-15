from CustomTranformers.Custom_Transf import get_best_match
import numpy as np 
import json 
import pandas as pd 
import os 
########################################
# Input validation functions
with open(os.path.join('pickles', 'columns.json')) as fh:
    columns = json.load(fh)
with open(os.path.join('pickles', 'boundriesOfNumeric.json')) as fh:
    Bounders = json.load(fh)
with open(os.path.join('pickles', 'categories_values.json')) as fh:
    categories_values = json.load(fh)
def check_request(request):
    """
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    
    if "observation_id" not in request:
        error = "Field `observation_id` missing from request: {}".format(request)
        
        return False, error 
    id = request.get('observation_id')
    if pd.isna(id) or not id : 
        error = "Field `observation_id` must have valid unique value"
        return False, error
    
    
    return True, "" 

def check_True_outcome(observation) : 
    if "outcome" not in observation:
        error = "Field `outcome` missing "
        return False, error 
    outcome = observation.get('outcome')
    if isinstance(outcome, bool):
        return True ,""
    else : 
        if pd.isna(outcome) or not outcome:
                return False ,"U have enter None value to outcome and it should be boolen [true , false]"
        else : 
            
            if outcome in ['true','True','False','false'] :  
            
                error = "Field `outcome` u have enter good value but should be in boolen not string"
                return False, error
            
            else : 
                error = "Field `outcome` should be boolen get only true or false "
                return False, error
    
def check_valid_update(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_columns = set(['observation_id','outcome'])
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    return True, ""
    
    
    
    
def check_valid_column(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_columns = set(columns)
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    return True, ""



def check_categorical_values(observation):
    """
        Validates that all categorical fields are in the observation and values are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_category_map = categories_values.copy()
    legislation_na = False
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value in valid_categories:
                continue 
            else :
                if pd.notna(value) and value :
                    possible = get_best_match(value , valid_categories) 
                    if possible : 
                        error = "the value provided for {}: {}. is close to `{} ` may be u misspelling it ".format(
                        key, value,possible)
                        return False, error , legislation_na
                    else : 
                        if key not in ['Legislation' ,'station','Object of search']:

                            error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                                key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                            return False, error , legislation_na
                        else : 
                                continue 
                            
                else : 
                    if key not in ['Legislation']:
                       
                        error = "invalid nan input for {}:".format(key)
                        return False, error , legislation_na
                    else :
                        legislation_na = True 
                        continue
                    
        else:
            error = f"Categorical field {key} missing"
            return False, error

    return True, "" , legislation_na

def check_date(observation):
    if "Date" not in observation:
        error = "Field `Date` missing "
        return False, error 
    date = observation.get('Date')
    if pd.isna(date) or not date : 
        error = "Field `date` has missing value"
        return False, error

    try:
        pd.to_datetime(date,infer_datetime_format=True)
        return True ,""
    except ValueError:
        error = "Field `Date` has invalid date"
        return False , error

def check_Poperation(observation):
    if "Part of a policing operation" not in observation:
        error = "Field `Part of a policing operation` missing "
        return False, error 
    Poperation = observation.get('Part of a policing operation')
    if isinstance(Poperation, bool):
        return True ,""
    else : 
        if pd.isna(Poperation) or not Poperation:
                return True ,"it was nan but we fill it as False, try fill it to for better outcome"
        else : 
            
            if Poperation in ['true','True','False','false'] :  
            
                error = "Field `Part of a plocinig operation` u have enter good value but should be in boolen not string"
                return False, error
            
            else : 
                error = "Field `Part of a plocinig operation` should be boolen get only true or false "
                return False, error
    

def check_long(observation):
    """
        Validates that observation contains valid hour value 
        
        Returns:
        - assertion value: True if hour is valid, False otherwise
        - error message: empty if hour is valid, False otherwise
    """
    if "Longitude" not in observation:
        error = "Field `Longitude` missing "
        return False, error 
    long = observation.get("Longitude")
        
    if pd.isna(long) or not long:
        
        return True ,"it was nan but we fill it as using the station field, try fill it to for better outcome"

    if not isinstance(long, float):
        error = "Field `Longitude` must be float value "
        return False, error
    maximum = Bounders['Longitude']['max']
    minimum = Bounders['Longitude']['min']
    if long < minimum or long > maximum:
        error = f"Field `Longitude` can't be {long}, should be between {minimum} and {maximum}"
        return False, error

    return True, ""
def check_lat(observation):
    """
        Validates that observation contains valid hour value 
        
        Returns:
        - assertion value: True if hour is valid, False otherwise
        - error message: empty if hour is valid, False otherwise
    """
    if "Latitude" not in observation:
        error = "Field `Latitude` missing "
        return False, error 
    lat = observation.get("Latitude")
        
    if pd.isna(lat) or not lat:
        
        return True ,"it was nan but we fill it as using the station field, try fill it to for better outcome"

    if not isinstance(lat, float):
        error = "Field `Latitude` must be float value "
        return False, error
    maximum = Bounders['Latitude']['max']
    minimum = Bounders['Latitude']['min']
    if lat < minimum or lat > maximum:
        error = f"Field `Latitude` can't be {lat}, should be between {minimum} and {maximum}"
        return False, error

    return True, ""



# End input validation functions
########################################
