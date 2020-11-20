========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        |
    * - package
      - | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/nmvae/badge/?style=flat
    :target: https://readthedocs.org/projects/nmvae
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/wkopp/nmvae.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/wkopp/nmvae

.. |commits-since| image:: https://img.shields.io/github/commits-since/wkopp/nmvae/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/wkopp/nmvae/compare/v0.0.0...master



.. end-badges

Collection of variational autoencoders

* Free software: GNU Lesser General Public License v3 or later (LGPLv3+)

Installation
============

::

    pip install nmvae

You can also install the in-development version with::

    pip install https://github.com/wkopp/nmvae/archive/master.zip


Documentation
=============


https://nmvae.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
