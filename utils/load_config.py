import json
def load_config_from_json(json_filename=None):
    if json_filename is None:
        json_filename = './configs/test_segmentation_config.json'
    config = json.load(open(json_filename))
    return config
    
