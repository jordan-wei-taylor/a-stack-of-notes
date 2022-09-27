from   matplotlib import pyplot as plt

import os


def savefig(path, *args, **kwargs):
    directory = os.path.dirname(path)
    filename  = os.path.basename(path)
    assert len(directory)

    directory = f'../a-stack-of-notes/_assets/{directory}'

    os.makedirs(directory, exist_ok = True)

    if 'dpi' in kwargs:
        dpi = kwargs.pop('dpi')
    else:
        dpi = 500
    plt.savefig(f'{directory}/{filename}', *args, dpi = dpi, **kwargs)
