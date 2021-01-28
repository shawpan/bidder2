import numpy as np

def get_types_of_attributes():
    return {
        'deliveryid': np.string_,
        'dayofweek': np.uint8,
        'hour': np.uint8,
        'pub_sspid': np.string_,
        'pub_accountid': np.string_,
        'pub_as_siteid': np.string_,
        'pub_as_adspaceid': np.string_,
        'pub_as_domain': np.string_,
        'pub_as_pageurl': np.string_,
        'pub_as_dimensions': np.string_,
        'pub_as_viewrate': np.float16,
        'pub_as_position': np.string_,
        'pub_as_caps': np.string_,
        'req_buymodel': np.string_,
        'req_auctiontype': np.string_,
        'device_os': np.string_,
        'device_model': np.string_,
        'rtb_ctr': np.float16,
        'rtb_viewrate': np.float16,
        'rtb_bidfloor': np.float16,
        'rtb_battr': np.string_,
        'rtb_tagid': np.string_,
        'user_ip': np.string_,
        'user_market': np.string_,
        'user_city': np.string_,
        'ad_imptype': np.string_,
        'user_id': np.string_,
        'pub_as_iabcategoryid': np.string_,
        'req_bid': np.float16,
        'price': np.float16,
        'advcostcpm': np.float16,
        'won': np.uint8,
        'targetbid': np.float16,
        'click': np.uint8,
        'imp': np.uint8,
        'imp_0': np.uint8,
        'imp_1': np.uint8,
        'imp_2': np.uint8,
        'imp_3': np.uint8,
        'imp_4': np.uint8,
        'imp_5': np.uint8,
        'imp_6': np.uint8,
        'imp_7': np.uint8,
        'imp_8': np.uint8,
        'imp_9': np.uint8,
        'imp_10': np.uint8,
        'imp_11': np.uint8,
        'imp_12': np.uint8,
        'imp_13': np.uint8,
        'imp_14': np.uint8
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
