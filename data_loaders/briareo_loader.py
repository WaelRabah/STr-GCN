import json

def load_Briareo_data():
    def load_json(filename):
        with open(filename) as f :
            # returns JSON object as 
            # a dictionary
            data = json.load(f)
            return data
            
    train_data=load_json("./data/Briareo_data/train.json")
    validation_data=load_json("./data/Briareo_data/validation.json")
    test_data=load_json("./data/Briareo_data/test.json")

    return train_data, validation_data, test_data