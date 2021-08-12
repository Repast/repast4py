# Development Notes #

## Requirements

* Python 3.7+
* mpi4py
* PyTorch
* NumPy >= 1.18
* nptyping (`pip install nptyping`)
* numba
* typing-extensions if < 3.8
* pyyaml


## Linting ##

flake8 - configuration, exclusions etc. are in setup.cfg

## Documentation Guidelines ##

* Use Sphinx and readthedocs.io

https://docs.python-guide.org/writing/documentation/#sphinx

* Use restructured text:

https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

* Use google style doc strings:

https://www.sphinx-doc.org/en/1.8/usage/extensions/example_google.html?highlight=docstring

* Pep 484 type hints - when these are present then they do not need to be included in the doc string

## Generating API Docs ##

If a new module is added then, from within the docs directory `sphinx-apidoc -e -o source ../src/repast4py` 
to generate the rst for that module.

And `make html` to create the html docs.

`make clean` followed by `make html` will build from scratch.


## Generating ASCIIDoc Manual ##

### Prerequisites ###

1. Install asciidoctor.
  * With apt (Ubuntu): `sudo apt-get install -y asciidoctor`
  * Other OSes, see `https://docs.asciidoctor.org/asciidoctor/latest/install/`
2. Install pygments for code syntax highlighting.
  * Ubuntu: `gem install --user-install pygments.rb`
  * Other OSes, see `https://docs.asciidoctor.org/asciidoctor/latest/syntax-highlighting/pygments/`

Currently (08/09/2021) the docs are generated as single html page. If we want multiple
pages, see `https://github.com/owenh000/asciidoctor-multipage`

### Generating the Docs ###

```
cd docs/guide
asciidoctor user_guilde.adoc
```

This generates a user_guide.html that can be viewed in a browser.

## Multi Process Seg Fault Debugging

See https://www.open-mpi.org/faq/?category=debugging#serial-debuggers for using gdb with mpi.

General idea is to add this code

```
  {
         volatile int i = 0;
         char hostname[256];
         gethostname(hostname, sizeof(hostname));
         printf("PID %d on %s ready for attach\n", getpid(), hostname);
         fflush(stdout);
         while (0 == i)
             sleep(5);
    }
```

to the module code function call that's triggering the segfault, and follow the directions in the link above. Note that might have to run gdb via sudo.