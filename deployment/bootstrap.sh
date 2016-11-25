#!/bin/sh

ansible $@ --sudo -m raw -a "apt-get install -y python-simplejson"
