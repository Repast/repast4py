# Development Notes #

## Requirements

* Python 3.7+
* mpi4py
* PyTorch
* NumPy
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

Install asciidoc3 using pip: https://www.asciidoc3.org/pypi.html

To generate the user guide in html:

`python ~/asciidoc3/asciidoc3.py -a toc -a toclevel=3 --backend=html5 -a icons -a iconsdir=~/asciidoc3/images/icons user_guide.adoc`

The guide is written in asciidoc format.

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