# deep_sentence deployment

## Setup

Copy `ansible.cfg.example` to `ansible.cfg` and customize to your needs.

Then, install dependencies

```
pip install -r requirements.txt
ansible-galaxy install -r requirements.yml
```

## Setup node

To setup a node, add it to `inventory/production` and run

```
ansible-playbook site.yml
```

If the node does not have Python 2 installed, you will need to run

```
./bootstrap.sh all
```

first.

## Deploy scraper

```
fab deploy
```
