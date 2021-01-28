import tensorflow as tf

def first_price_auction_loss_clm(labels, logits, features):
    # http://www0.cs.ucl.ac.uk/staff/w.zhang/rtb-papers/win-price-pred.pdf
    # P = predicted bid
    # L = lost bid
    # W = won bid
    # T = true winning bid
    # Et = T - B, E = P - B
    wons = tf.reshape(labels[:, 1], [-1, 1])
    dist = tf.distributions.Normal(loc=0.0, scale=1.0)
    
    target_bids = labels[:, 0]
    predicted_bids = tf.exp(logits[:, 0])

    e = tf.reshape((predicted_bids - target_bids), [-1,1])
    
    # prevent log(0)
    e = e + 1e-8
    
    # error on W
    error_on_W = -dist.log_prob(-e)
    # error on L
    error_on_L = -dist.log_cdf(e)
    
    bid_error_on_lost = error_on_L * (1. - wons)
    bid_error_on_won = error_on_W * wons
    
    bid_error = bid_error_on_lost + bid_error_on_won
    # keep the bid error only and make won, 0/1 labels error 0
    bid_error = bid_error * [1, 0]
    
#     predicted_win_probabilty = tf.reshape(tf.sigmoid(logits[:,1]), [-1,1])
#     win_error_on_won = wons * tf.log(predicted_win_probabilty)
#     win_error_on_lost = (1. - wons) * tf.log(1. - predicted_win_probabilty)
    
#     win_error = -(win_error_on_won + win_error_on_lost)
#     win_error = win_error * [0, 1]
    
#     with tf.control_dependencies([tf.compat.v1.assert_non_negative(bid_error), tf.compat.v1.assert_non_negative(win_error)]):
#         error = (bid_error + win_error)
    with tf.control_dependencies([tf.compat.v1.assert_non_negative(bid_error)]):
        error = bid_error
    
#     sess = tf.compat.v1.Session()
#     with sess.as_default():
#         print_op = tf.print("labels: ", labels, "logits: ", logits, "wons: ", wons, "target_bids:", target_bids , "bid_error: ", bid_error, "win_error: ", win_error, "error: ", error, "predicted_bids: ", predicted_bids, "predicted_win_probabilty: ", predicted_win_probabilty)
#         with tf.control_dependencies([print_op]):
#             l = tf.zeros([3, 2])

    return error

def first_price_auction_loss_left_censored(labels, logits, features):
    # P = predicted bid
    # L = lost bid
    # W = won bid
    # T = true winning bid
    # Et = T - B, E = P - B
    wons = tf.reshape(labels[:, 1], [-1, 1])
    dist = tf.distributions.Normal(loc=0.0, scale=1.0)
    
    target_bids = labels[:, 0]
    predicted_bids = tf.exp(logits[:, 0])

    e = tf.reshape((predicted_bids - target_bids), [-1,1])
    # P < L
    error_of_P_less_than_L = tf.square(tf.clip_by_value(e, tf.float32.min, 0.))
    # P(Et <= E)
    error_of_probability_for_Et_less_than_equal_to_E_on_L = -dist.log_cdf(tf.clip_by_value(e, 0., tf.float32.max))

    # (P - W)^2
    error_on_W = tf.square(e)
    
    bid_error_on_lost = (error_of_P_less_than_L + error_of_probability_for_Et_less_than_equal_to_E_on_L) * (1. - wons)
    bid_error_on_won = error_on_W * wons
    
    bid_error = bid_error_on_lost + bid_error_on_won
    bid_error = bid_error * [1, 0]
    
#     predicted_win_probabilty = tf.reshape(tf.sigmoid(logits[:,1]), [-1,1])
#     win_error_on_won = wons * tf.log(predicted_win_probabilty)
#     win_error_on_lost = (1. - wons) * tf.log(1. - predicted_win_probabilty)
    
#     win_error = -(win_error_on_won + win_error_on_lost)
#     win_error = win_error * [0, 1]
    
#     with tf.control_dependencies([tf.compat.v1.assert_non_negative(bid_error), tf.compat.v1.assert_non_negative(win_error)]):
#         error = (bid_error + win_error)
    with tf.control_dependencies([tf.compat.v1.assert_non_negative(bid_error)]):
        error = bid_error
    
#     sess = tf.compat.v1.Session()
#     with sess.as_default():
#         print_op = tf.print("labels: ", labels, "logits: ", logits, "wons: ", wons, "target_bids:", target_bids , "bid_error: ", bid_error, "win_error: ", win_error, "error: ", error, "predicted_bids: ", predicted_bids, "predicted_win_probabilty: ", predicted_win_probabilty)
#         with tf.control_dependencies([print_op]):
#             l = tf.zeros([3, 2])

    return error

def first_price_auction_loss_left_right_censored(labels, logits, features):
    # P = predicted bid
    # L = lost bid
    # W = won bid
    # T = true winning bid
    # Et = T - B, E = P - B
    wons = tf.reshape(labels[:, 1], [-1, 1])
    dist = tf.distributions.Normal(loc=0.0, scale=1.0)
    
    target_bids = labels[:, 0]
    predicted_bids = tf.exp(logits[:, 0])

    e = tf.reshape((predicted_bids - target_bids), [-1,1])
    # P < L
    errorOfPlessThanL = tf.exp(-tf.clip_by_value(e, tf.float32.min, 0.))
    # P(Et <= E)
    errorOfProbabilityForEtLessThanEqualToEOnL = -dist.log_cdf(tf.clip_by_value(e, 0., tf.float32.max))

    # P > W
    errorOfPgreaterThanW = tf.clip_by_value(e, 0., tf.float32.max)
    # P(Et <= E)
    errorOfProbabilityForEtLessThanEqualToEOnW = -dist.log_cdf(-tf.clip_by_value(e, tf.float32.min, 0.))
    
    bid_error_on_lost = (errorOfPlessThanL + errorOfProbabilityForEtLessThanEqualToEOnL) * (1. - wons)
    bid_error_on_won = (errorOfPgreaterThanW + errorOfProbabilityForEtLessThanEqualToEOnW) * wons
    
    bid_error = bid_error_on_lost + bid_error_on_won
    bid_error = bid_error * [1, 0]
    
    predicted_win_probabilty = tf.reshape(tf.sigmoid(logits[:,1]), [-1,1])
    win_error_on_won = wons * tf.log(predicted_win_probabilty)
    win_error_on_lost = (1. - wons) * tf.log(1. - predicted_win_probabilty)
    
    win_error = -(win_error_on_won + win_error_on_lost)
    win_error = win_error * [0, 1]
    
    with tf.control_dependencies([tf.compat.v1.assert_non_negative(bid_error), tf.compat.v1.assert_non_negative(win_error)]):
        error = (bid_error + win_error)
    
#     sess = tf.compat.v1.Session()
#     with sess.as_default():
#         print_op = tf.print("labels: ", labels, "logits: ", logits, "wons: ", wons, "target_bids:", target_bids , "bid_error: ", bid_error, "win_error: ", win_error, "error: ", error, "predicted_bids: ", predicted_bids, "predicted_win_probabilty: ", predicted_win_probabilty)
#         with tf.control_dependencies([print_op]):
#             l = tf.zeros([3, 2])

    return error