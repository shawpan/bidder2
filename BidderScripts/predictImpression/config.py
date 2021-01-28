import json
import numpy as np
import os

CONFIG = None
CONFIG_FILE = None

"""
Get the configurations from config.json file as object
"""
def get_config():
    global CONFIG
    with open(os.getenv('SM_HP_CONFIG_FILE', default='config.json'), 'r') as f:
        CONFIG = json.load(f)
    return CONFIG
#2|12440|8084|23678|null|null|300x250|0.707278907|null|15166603496|4|2|19930|100000|null|null|0.315631807|null|2152980677|84.241.195.0|75|Amsterdam|4|0.856686056|null|0|0|0.8566860556602478
def get_types_of_attributes():
    return {
        'deliveryid' : np.string_,          # 307219911482
        'dayofweek' : np.uint8,              # 3
        'hour' : np.uint8,              # 3
        'pub_sspid' : np.string_,           # 2
        'pub_as_adspaceid' : np.string_,    # 23678
        'pub_as_domain' : np.string_,   # 300x250
        'pub_as_dimensions' : np.string_,   # 300x250
        'pub_as_position' : np.string_,   # 300x250
        'pub_as_viewrate' : np.float16,   # 300x250
        'device_os' : np.string_,           # 19930
        'device_model' : np.string_,        # 100000
        'user_ip' : np.string_,             # 84.241.195.0
        'user_market' : np.string_,         # 75
        'user_city' : np.string_,           # Amsterdam
        'ad_imptype' : np.string_,           # Amsterdam
        'user_id' : np.string_,          # 4
        'ad_formatid': np.string_,
        'pub_as_iabcategoryid': np.string_,
        'req_auctiontype' : np.string_,
        'imp_0' : np.uint8,
        'imp_1' : np.uint8,
        'imp_2' : np.uint8,
        'imp_3' : np.uint8,# 0
        'imp_4' : np.uint8,
        'imp_5' : np.uint8,# 0
        'imp_6' : np.uint8,
        'imp_7' : np.uint8,# 0
        'imp_8' : np.uint8,
        'imp_9' : np.uint8,# 0
        'imp_10' : np.uint8,
        'imp_11' : np.uint8,# 0
        'imp_12' : np.uint8,
        'imp_13' : np.uint8,# 0
        'imp_14' : np.uint8,# 0
        'dayofweek_hour' : np.uint8,              # 3
        'domain_position' : np.string_,     # null
        'weight' : np.float16                 # 0
    }

def get_default_values_for_csv_columns():
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

# print(get_default_values_for_csv_columns())
