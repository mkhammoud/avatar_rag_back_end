def list_dict_sum(list_dict):
    sum = {}
    for d in list_dict:
        for key, value in d.items():
            if key in sum:
                sum[key] += value
            else:
                sum[key] = value
    return sum

