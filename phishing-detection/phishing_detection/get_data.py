import sys
import utils
def load_data(path):
    """
    Load train, test, and validation phishing data.

    Returns:
        Tuple of raw x and y data for train, test, and validation sets.
    """
    # TODO handle error if it doesn't exist. Maybe not needed if dvc is used

    train = [line.strip() for line in open(f"{path}/train.txt", "r").readlines()[1:]]
    raw_X_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    test = [line.strip() for line in open(f"{path}/test.txt", "r").readlines()]
    raw_X_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    val=[line.strip() for line in open(f"{path}/val.txt", "r").readlines()]
    raw_X_val=[line.split("\t")[1] for line in val]
    raw_y_val=[line.split("\t")[0] for line in val]

    return raw_X_train, raw_y_train, raw_X_val, raw_y_val, raw_X_test, raw_y_test


def main():
    path = sys.argv[1]
    raw_X_train, raw_y_train, raw_X_val, raw_y_val, raw_X_test, raw_y_test = load_data(path)

    utils.save_data_as_text(raw_X_train, f"{path}/raw/X_train.txt")
    utils.save_data_as_text(raw_y_train, f"{path}/raw/y_train.txt")
    utils.save_data_as_text(raw_X_val, f"{path}/raw/X_val.txt")
    utils.save_data_as_text(raw_y_val, f"{path}/raw/y_val.txt")
    utils.save_data_as_text(raw_X_test, f"{path}/raw/X_test.txt")
    utils.save_data_as_text(raw_y_test, f"{path}/raw/y_test.txt")

if __name__ == "__main__":
    main()