#!/bin/sh

set -e

export FLASK_APP=app
export FLASK_ENV=production

teardown() {
  cd ../..
}

trap_ctrlc() {
  teardown
  exit 2
}

trap "trap_ctrlc" 2

cd src/client

if [ $# -eq 0 ]
  then
    flask run
  else
    if echo $* | grep -e "--watch" -q
    then
      nodemon -e "py" --exec "flask run"
    else
      flask run
    fi
fi

teardown
