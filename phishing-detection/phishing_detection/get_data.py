"""
Provides functions to load the data.

"""

def load_data():
    """
    Load train, test, and validation phishing data.

    Returns:
        Tuple of raw x and y data for train, test, and validation sets.
    """
    # TODO handle error if it doesn't exist. Maybe not needed if dvc is used
    # TODO: fix the sources

    train = [line.strip() for line in open("data/train.txt", "r").readlines()[1:]]
    raw_X_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    test = [line.strip() for line in open("data/test.txt", "r").readlines()]
    raw_X_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    val=[line.strip() for line in open("data/val.txt", "r").readlines()]
    raw_X_val=[line.split("\t")[1] for line in val]
    raw_y_val=[line.split("\t")[0] for line in val]

    return raw_X_train, raw_y_train, raw_X_val, raw_y_val, raw_X_test, raw_y_test
