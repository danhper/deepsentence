from fabric.api import env, run, cd


env.user = 'deep_sentence'
env.hosts = ['scraper.internal.deepsentence.com']
env.use_ssh_config = True

CODE_DIR  = '/home/deep_sentence/deep_sentence'
BRANCH = 'master'


def deploy():
    with cd(CODE_DIR):
        run("git fetch")
        sha = run("git rev-parse origin/{0}".format(BRANCH))
        run("git checkout {0}".format(sha))
        run("python setup.py install --user")
        run("scrapyd-deploy")
