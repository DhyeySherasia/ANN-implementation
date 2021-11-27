import yaml

def read_config(config_paht):
    with open(config_paht) as config_file:
        content = yaml.safe_load(config_file)
    
    return content