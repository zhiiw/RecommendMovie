import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
def create_utility_matrix(data, formatizer={'user': 0, 'item': 1, 'value': 2}):
    """
        :param data:      Array-like, 2D, nx3
        :param formatizer:pass the formatizer
        :return:          utility matrix (n x m), n=users, m=items
    """

    itemField = formatizer['item']
    userField = formatizer['user']
    valueField = formatizer['value']
    userList = data.iloc[:, userField].tolist()
    itemList = data.iloc[:, itemField].tolist()
    valueList = data.iloc[:, valueField].tolist()
    users = list(set(data.iloc[:, userField]))
    items = list(set(data.iloc[:, itemField]))
    users_index = {users[i]: i for i in range(len(users))}
    pd_dict = {item: [np.nan for i in range(len(users))] for item in items}
    for i in range(0, len(data)):
        item = itemList[i]
        user = userList[i]
        value = valueList[i]
        pd_dict[item][users_index[user]] = value
        X = pd.DataFrame(pd_dict)
    X.index = users

    itemcols = list(X.columns)
    items_index = {itemcols[i]: i for i in range(len(itemcols))}
    # users_index gives us a mapping of user_id to index of user
    # items_index provides the same for items
    return X, users_index, items_index


def svd(train, k):
    utilMat = np.array(train)    # the nan or unavailable entries are masked
    mask = np.isnan(utilMat)
    masked_arr = np.ma.masked_array(utilMat, mask)
    item_means = np.mean(masked_arr, axis=0)    # nan entries will replaced by the average rating for each item
    utilMat = masked_arr.filled(item_means)
    x = np.tile(item_means, (utilMat.shape[0],1))    # we remove the per item average from all entries.
    # the above mentioned nan entries will be essentially zero now
    utilMat = utilMat - x    # The magic happens here. U and V are user and item features
    U, s, V=np.linalg.svd(utilMat, full_matrices=False)
    s=np.diag(s)    # we take only the k most significant features
    s=s[0:k,0:k]
    U=U[:,0:k]
    V=V[0:k,:]
    s_root=sqrtm(s)
    Usk=np.dot(U,s_root)
    skV=np.dot(s_root,V)
    UsV = np.dot(Usk, skV)
    UsV = UsV + x
    print("svd done")
    return UsV