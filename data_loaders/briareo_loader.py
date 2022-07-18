import json

def load_Briareo_data():
    def load_json(filename):
        with open(filename) as f :
            # returns JSON object as 
            # a dictionary
            data = json.load(f)
            return data
            
    train_data=load_json("./data/Briareo/train.json")
    validation_data=load_json("./data/Briareo/validation.json")
    test_data=load_json("./data/Briareo/test.json")

    return train_data, validation_data, test_data