
import yaml


def parse_yaml(filepath: str):

    with open(filepath) as fp:
        yaml_dict = yaml.safe_load(fp)
        
    return yaml_dict