import os

from log import _P, _L, _S


def wget(url, path):
    """
    Download url from the web

    Args:
        url (string) : url to download from the web
        path (string) : path where the dataset will be downloaded
    """

    # check if the path exist or not
    os.makedirs(os.path.normpath(path), exist_ok=True)
    if not os.path.exists(os.path.join(path, url.split('/')[-1])):
        os.system("wget {} -P {}".format(url, path))

def unzip(zipped, path):
    """
    Extract the zipped file in the path

    Args:
        zipped (string): file to unzip
        path (string): where to unzip
    """
    os.system("unzip -o {} -d {} | pv -l >/dev/null".format(zipped, path))
