"""
Provides functions to save and load data.

"""
import json


def save_data_as_text(data, file_path):
    """
    Save data to a text file.

    Args:
        data (List[str]): Data to be saved.
        file_path (str): Path to the output text file.
    """
    with open(file_path, "w", encoding="UTF-8") as file:
        for item in data:
            file.write(f"{item}\n")


def load_data_from_text(file_path):
    """
    Load data from a text file.

    Args:
        file_path (str): Path to the input text file.

    Returns:
        List[str]: Data loaded from the text file.
    """
    with open(file_path, "r", encoding="UTF-8") as file:
        data = [line.strip() for line in file.readlines()]
    return data



def save_json(char_index, file_path):
    """
    Save the char_index dictionary to a JSON file.

    Args:
        char_index (dict): Dictionary containing character-to-index mapping.
        file_path (str): Path to the output JSON file.
    """
    with open(file_path, 'w', encoding="UTF-8") as f:
        json.dump(char_index, f)

def load_json(file_path):
    """
    Load the char_index dictionary from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Loaded char_index dictionary.
    """
    with open(file_path, 'r', encoding="UTF-8") as f:
        return json.load(f)
