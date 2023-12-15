#%%
def resolution_calculator(ratio="3:4", shorter=None, longer=None):
    """
    Calculate the width and height of an image based on a ratio.
    :param ratio: The ratio to calculate the width and height from.
    :param width: The width of the image.
    :param height: The height of the image.
    :return: The width and height of the image based on the ratio.
    """
    if ratio == "3:4" or ratio == "4:3":
        if shorter is not None:
            return shorter, float(shorter * 4 / 3)
        elif longer is not None:
            return float(longer * 3 / 4), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    elif ratio == "9:16" or ratio == "16:9":
        if shorter is not None:
            return shorter, float(shorter * 16 / 9)
        elif longer is not None:
            return float(longer * 9 / 16), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    elif ratio == "1:1":
        if shorter is not None:
            return shorter, shorter
        elif longer is not None:
            return longer, longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    elif ratio == "2:3" or ratio == "3:2":
        if shorter is not None:
            return shorter, float(shorter * 3 / 2)
        elif longer is not None:
            return float(longer * 2 / 3), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    elif ratio == "5:7" or ratio == "7:5":
        if shorter is not None:
            return shorter, float(shorter * 7 / 5)
        elif longer is not None:
            return float(longer * 5 / 7), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    elif ratio == "4:5" or ratio == "5:4":
        if shorter is not None:
            return shorter, float(shorter * 5 / 4)
        elif longer is not None:
            return float(longer * 4 / 5), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    elif ratio == "1:2" or ratio == "2:1":
        if shorter is not None:
            return shorter, float(shorter * 2 / 1)
        elif longer is not None:
            return float(longer * 1 / 2), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    elif ratio == "2.2:1" or ratio == "1:2.2":
        if shorter is not None:
            return shorter, float(shorter * 2.2 / 1)
        elif longer is not None:
            return float(longer * 1 / 2.2), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    elif ratio == "12:5" or ratio == "5:12":
        if shorter is not None:
            return shorter, float(shorter * 12 / 5)
        elif longer is not None:
            return float(longer * 5 / 12), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    elif ratio == "21:9" or ratio == "9:21":
        if shorter is not None:
            return shorter, float(shorter * 21 / 9)
        elif longer is not None:
            return float(longer * 9 / 21), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    elif ratio == "4:1" or ratio == "1:4":
        if shorter is not None:
            return shorter, float(shorter * 4 / 1)
        elif longer is not None:
            return float(longer * 1 / 4), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    elif ratio == "5:3" or ratio == "3:5":
        if shorter is not None:
            return shorter, float(shorter * 5 / 3)
        elif longer is not None:
            return float(longer * 3 / 5), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    elif ratio == "1.6180:1" or ratio == "1:1.6180" or ratio == "golden":
        if shorter is not None:
            return shorter, float(shorter * 1.6180 / 1)
        elif longer is not None:
            return float(longer * 1 / 1.6180), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")    
    elif ratio == "2.4142:1" or ratio == "1:2.4142" or ratio == "silver":
        if shorter is not None:
            return shorter, float(shorter * 2.4142 / 1)
        elif longer is not None:
            return float(longer * 1 / 2.4142), longer
        else:
            raise ValueError("Either shorter or longer must be specified.")
    else:
        raise ValueError("Invalid ratio specified.")
    
resolution_calculator(ratio="silver", longer=5.6)
# %%
