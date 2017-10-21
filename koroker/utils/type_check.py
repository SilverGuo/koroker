# check if list
def check_list(l):
    return type(l) == list


# check if str
def check_str(s):
    return type(s) == str


# check if string list
def check_string_list(lstr):
    if not check_list(lstr):
        return False
    for s in lstr:
        if not check_str(s):
            return False
    return True
