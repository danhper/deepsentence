from fabric.api import env, run, cd, shell_env
from fabric.contrib import files


env.user = 'deep_sentence'
env.hosts = ['internal.deepsentence.com']
env.use_ssh_config = True

CODE_DIR  = '/home/deep_sentence/deep_sentence'
BRANCH = 'master'


def deploy():
    with cd(CODE_DIR):
        run("git fetch")
        sha = run("git rev-parse origin/{0}".format(BRANCH))
        run("git checkout {0}".format(sha))

        # install with both pip2 and pip3 to be able to use both jupyter kernels
        run("pip2 install -r requirements.txt --user")
        run("pip3 install -r requirements.txt --user")

        run("python3 setup.py install --user")

        # scrapyd only supports python2
        run("python2 setup.py install --user")
        run("scrapyd-deploy")

        if files.exists('tmp/webapp.pid'):
            run("kill -HUP $(cat tmp/webapp.pid)")

        run("make webapp_setup")


def download_models():
    with cd(CODE_DIR):
        with shell_env(PROJECT_ROOT=CODE_DIR):
            run("make download_models")
