#!/bin/bash

set -eu

REPO=$HOME/Documents/repos/repast4py.site
THIS=$( cd $( dirname $0 ) ; /bin/pwd )
ROOT=$( cd ../../  ; /bin/pwd )

WEBSITE=https://repast.github.io/repast4py.site

# Update the landing page
echo "Building the landing page"
asciidoctor -a website=$WEBSITE landing.adoc -o $REPO/index.html


echo "Building the user guide"
mkdir -p $REPO/guide/images
asciidoctor -a website=$WEBSITE $THIS/../guide/user_guide.adoc -o $REPO/guide/user_guide.html
cp $THIS/../guide/images/* $REPO/guide/images

echo "Building API Docs"
mkdir -p $REPO/apidoc
cd $THIS/../..
CC=mpicxx CXX=mpicxx python setup.py build_ext --inplace
cd $THIS/..
make clean
make html
cp -r _build/html/* $REPO/apidoc/

echo "Building Examples"
mkdir -p $REPO/examples
cd $THIS/../../examples
asciidoctor -a website=$WEBSITE examples.adoc
cp examples.html $REPO/examples/examples.html

adocs=("examples/rumor/rumor_model.adoc" 
       "examples/rndwalk/random_walk.adoc"
       "examples/zombies/zombies.adoc"
      )
for f in "${adocs[@]}"
do
    path=$ROOT/$f
    # pd=$( dirname $path)
    # cd $pd
    # echo $pd
    asciidoctor -a website=$WEBSITE $path
done

cd $ROOT
rm -f $REPO/examples/repast4py_example_models.zip 
zip $REPO/examples/repast4py_example_models.zip -r examples -i@$THIS/examples_to_include


# cd $REPO
# git commit -a -m "Updated with latest"
# git push origin main