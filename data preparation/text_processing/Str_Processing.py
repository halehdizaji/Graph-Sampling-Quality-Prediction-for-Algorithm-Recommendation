import ast

def cnvrt_str_list_to_int_list(str_list):
    res = ast.literal_eval(ini_list)

    # printing final result and its type
    print("final list", res)
    print(type(res))