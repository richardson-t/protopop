from os import path
from subprocess import run
from shutil import move, unpack_archive

datapath = path.dirname(__file__)
record_id = None  # change when you make it


def retrieve_tracks():
    """
    Retrieve the template track information from the 
    `host Zenodo repository <{link}>`_. Includes protostellar 
    evolutionary tracks and corresponding flux evolutionary tracks.
    """
    if path.exists(f'{datapath}/../track_data'):
        raise RuntimeError(
            'track data files already exist; move or delete them in order to retrieve files from the repository')

    fn = 'track_data.tar.xz'

    run(f'uvx zenodo_get {record_id} -g {fn}', shell=True)
    unpack_archive(f'{fn}')
    move('track_data', f'{datapath}/..')
    run(f'rm {fn}', shell=True)


def retrieve_clusters():
    """
    Retrieve premade cluster models from the 
    `host Zenodo repository <{link}>`_; 
    includes all clusters made for 
    `Richardson+ (in prep) <{link}>`_.
    """
    if path.exists(f'cluster_data'):
        raise RuntimeError(
            'cluster data files already exist; move or delete them in order to retrieve files from the repository')

    fn = 'cluster_data.tar.xz'

    run(f'uvx zenodo_get {record_id} -g {fn}', shell=True)
    unpack_archive(f'{fn}')
    run(f'rm {fn}', shell=True)


def retrieve_all_data():
    retrieve_clusters()
    retrieve_tracks()
