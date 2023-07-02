import os



def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_data_dir():
    return '/home/julen/data/code/data'
