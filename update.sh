#!/bin/bash

git checkout master
git add -A . && git commit -m "Upload"
git push origin master
