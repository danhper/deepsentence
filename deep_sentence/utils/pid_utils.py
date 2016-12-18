import os
import os.path as path

from deep_sentence import settings

def is_process_running(pid_file):
    if not path.isfile(pid_file):
        return False
    with open(pid_file, 'r') as f:
        pid = int(f.read())
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        if e.errno == 3:
            return False
        raise e

def get_pid_file(name):
    return path.join(settings.PROJECT_ROOT, 'tmp', "{0}.pid".format(name))

def check_and_write_pid(name):
    pid_file = get_pid_file(name)
    if is_process_running(pid_file):
        raise RuntimeError('process already running')
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

def clean_pid_file(name):
    pid_file = get_pid_file(name)
    if path.isfile(pid_file):
        os.remove(pid_file)
