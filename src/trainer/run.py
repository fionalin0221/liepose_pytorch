from .testbed import Testbed
import argparse
import yaml
from types import SimpleNamespace

# Function to "recursively" convert a dictionary into a SimpleNamespace objects
def dict_to_namespace(d):
    if isinstance(d, dict):
        # Convert all "nested" dictionaries to SimpleNamespace
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def get_configs(desc="Liepose"):
    """
    Parses command-line arguments to load a configuration file and converts it into a namespace object.

    Args:
    desc (str): The description of the experiment

    Returns:
    SimpleNamespace: Namespace object contatining configuration values from YAML file.
    """
    # Define the argument parser
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the YAML file and convert it to an object
    try:
        with open(args.config, 'r') as file:
            config_dict = yaml.safe_load(file)
            config = dict_to_namespace(config_dict)
            return config
    except FileNotFoundError:
        print(f"Error: The file '{args.config}' does not exist.")

    return 

# Main function to execute the training process
def main():

    # Parse the arguments and load configuration
    config = get_configs()

    testbed = Testbed(config)
    # testbed.train()
    # testbed.test()
    # testbed.visualize()
    testbed.visualize_video()


if __name__ == "__main__":
    main()

