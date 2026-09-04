from pathlib import Path
import tomllib

CONFIG_PATH = Path(__file__).parent.parent / 'config.toml'

def get_config(name):
    with CONFIG_PATH.open('rb') as file:
        return tomllib.load(file)[name]