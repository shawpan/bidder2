import json
import numpy as np
import os

__CONFIG__ = None


def get_config():
    """ Get the configurations
    Returns:
        __CONFIG__(dict): dictionary of configuration
    """
    global __CONFIG__
    with open(os.getenv('SM_HP_CONFIG_FILE', default='config.json'), 'r') as f:
        __CONFIG__ = json.load(f)
    return __CONFIG__


def get_types_of_attributes():
    """ Get the desired types of the columns
    Returns:
        types(dict): Dictionary of desired types of columns
    """
    return {
        'deliveryid': np.string_,          # 307219911482
        'dayofweek': np.uint8,              # 3
        'hour': np.uint8,              # 3
        'pub_sspid': np.string_,           # 2
        'pub_as_adspaceid': np.string_,    # 23678
        'pub_as_domain': np.string_,   # 300x250
        'pub_as_dimensions': np.string_,   # 300x250
        'pub_as_position': np.string_,   # 300x250
        'pub_as_viewrate': np.float16,   # 300x250
        'device_os': np.string_,           # 19930
        'device_model': np.string_,        # 100000
        'user_ip': np.string_,             # 84.241.195.0
        'user_market': np.string_,         # 75
        'user_city': np.string_,           # Amsterdam
        'user_id': np.string_,           # Amsterdam
        'pub_as_iabcategoryid': np.string_,           # Amsterdam
        'req_auctiontype': np.string_,
        'price': np.float16,
        'advcostcpm': np.float16,
        'won': np.float16,                  # 0
        'targetbid': np.float16,                  # 0
        'domain_position': np.string_
    }


def get_default_values_for_csv_columns():
    """ Get default values for columns
    Returns:
        default_values(dict): Dictionary of default values for columns
    """
    default_value_for_dtypes = {
        np.string_: "0",
        np.int_: 0,
        np.uint8: 0,
        np.float16: 0.0,
        np.double: 0.0
    }
    types_of_attributes = get_types_of_attributes()
    default_values = []
    conf = get_config()
    for column_name, dtype in types_of_attributes.items():
        if column_name == conf['WEIGHT_COLUMN']:
            default_values.append(1.0)
        else:
            default_values.append(default_value_for_dtypes[dtype])

    return default_values
