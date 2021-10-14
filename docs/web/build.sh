#!/bin/bash

set -eu

REPO=$HOME/Documents/repos/goes_bing
THIS=$( cd $( dirname $0 ) ; /bin/pwd )

# Update the landing page
echo "Building the landing page"
asciidoctor landing.adoc -o $REPO/index.html


echo "Building the user guide"
mkdir -p $REPO/guide/images
asciidoctor $THIS/../guide/user_guide.adoc -o $REPO/guide/user_guide.html
cp $THIS/../guide/images/* $REPO/guide/images

echo "Building API Docs"
mkdir -p $REPO/apidoc
cd $THIS/..
make clean
make html
cp -r _build/html/* $REPO/apidoc/


# cd $REPO
# git commit -a -m "Updated with latest"
# git push origin main