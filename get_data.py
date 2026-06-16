from os import path
from subprocess import run
from shutil import move, unpack_archive

if path.exists('protopop/data'):
    raise RuntimeError('data files already exist; move or delete them in order to retrieve files from the repository')

run('uvx zenodo_get {record_id}', shell=True) #put in record id once you make one
unpack_archive('data.tar.xz')
move('data', 'protopop')
run('rm data.tar.xz', shell=True)
