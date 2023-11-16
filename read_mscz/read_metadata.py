# README
# Phillip Long
# September 27, 2023

# Create an object that can read JSON files for musescore into a prettier, pythonic format.


# IMPORTS
##################################################
import json
from types import SimpleNamespace
##################################################


# METADATA CLASS
##################################################

class Metadata():

    # initializer
    def __init__(self, path: str):

        # load JSON file as dictionary
        with open(path, "r") as file:
            self.metadata_dict = json.load(fp = file)
        
        # set attributes
        for key in tuple(self.metadata_dict.keys()):
            setattr(self, key, self.metadata_dict[key])
            vars(self)[key] = json.loads(s = json.dumps(obj = self.metadata_dict[key]), object_hook = lambda attribute: SimpleNamespace(**attribute))
        
    # accessor functions
    def is_public_domain(self) -> bool:
        return self.data.is_public_domain

    # print out metadata in a pretty, readable way
    def print(self):
        self._print_dict_recursively(dictionary = self.metadata_dict)
    
    # use recursion to print out metadata dictionary
    def _print_dict_recursively(self, dictionary: dict, recursion_level: int = 0):
        prefix = "".join(" " for _ in range(2 * recursion_level))
        for key in tuple(dictionary.keys()):
            if type(dictionary[key]) is dict:
                print(prefix + f"- {key}")
                self._print_dict_recursively(dictionary = dictionary[key], recursion_level = recursion_level + 1)
            else:
                print(prefix + f"- {key}: {dictionary[key]}")

##################################################


# PARSING FUNCTIONS
##################################################

##################################################


# MAIN READ METADATA FUNCTION
##################################################

def read_metadata(path: str) -> Metadata:
    """Read a JSON file into a Metadata object.

    Parameters
    ----------
    path : str or Path
        Path to the MuseScore file to read.

    Returns
    -------
    :class:`read_metadata.Metadata`
        Converted Metadata object.

    """

    return Metadata(path = path)

##################################################