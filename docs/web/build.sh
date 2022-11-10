#!/bin/bash

set -eu


REPO=$HOME/Documents/repos/repast4py.site
THIS=$( cd $( dirname $0 ) ; /bin/pwd )
ROOT=$( cd ../../  ; /bin/pwd )

WEBSITE=https://repast.github.io/repast4py.site

help() {
    echo "$(basename "$0") [-l|u|a|e] - builds the repast4py documentation."
    echo "If no option is specified, then all the documentation is built."
    echo
    echo "Options:"
    echo "  l   build the landing page and macos mpi doc"
    echo "  u   build the user guide"
    echo "  a   build the API documentation"
    echo "  e   build the examples"
}

landing_page() {
    # Update the landing page
    echo "Building landing page and macos mpi doc"
    asciidoctor -a website=$WEBSITE landing.adoc -o $REPO/index.html
    asciidoctor -a website=$WEBSITE macos_mpi_install.adoc -o $REPO/macos_mpi_install.html

}

user_guide() {
    echo "Building user guide"
    mkdir -p $REPO/guide/images
    asciidoctor -a website=$WEBSITE $THIS/../guide/user_guide.adoc -o $REPO/guide/user_guide.html
    cp $THIS/../guide/images/* $REPO/guide/images
}

api_docs() {
    echo "Building API Docs"
    mkdir -p $REPO/apidoc
    cd $THIS/../..
    CC=mpicxx CXX=mpicxx python setup.py build_ext --inplace
    cd $THIS/..
    make clean
    make html
    cp -r _build/html/* $REPO/apidoc/
}

examples() {
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
}

# push() {
#     echo "Pushing $REPO"
#     # cd $REPO
#     # git commit -a -m "Updated with latest"
#     # git push origin main
# }

while getopts "hluae" option; do
   case $option in
        h) # display Help
            help
            exit;;
        l)
            landing_page
            ;;
        u)
            user_guide
            ;;
        a)
            api_docs
            ;;
        e)
            examples
            ;;
        *) # Invalid option
            echo "Error: Invalid option"
            help
            exit;;
   esac
done

if [ $OPTIND -eq 1 ]; then 
    landing_page
    user_guide
    api_docs
    examples
fi