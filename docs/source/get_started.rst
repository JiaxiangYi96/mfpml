.. image:: https://user-images.githubusercontent.com/25851824/200928507-a65327f9-bc70-4c12-beaa-bc6ba74d968e.svg
   :alt: logo
   :width: 60%
   :align: center


`Docs <https://bessagroup.github.io/bessa-pypi-template/>`__ | `GitHub <https://github.com/JiaxiangYi96/mfpml.git>`__


**First publication:** July 23, 2023

Installation
---------------

Building from source is required to work on a contribution (bug fix, new feature, code or documentation improvement).
We recommend using a `Linux distribution system <https://releases.ubuntu.com/focal/>`_.

.. _git_repo:

1. Use `Git <https://git-scm.com/>`_ to check out the latest source from the
   `mfpml repository <https://github.com/JiaxiangYi96/mfpml>`_ on
   Github.:

   .. code-block:: console

     git clone https://github.com/JiaxiangYi96/mfpml.git 
     cd mfpml


2. Install a recent version of Python (3.10)
   for instance using `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_.
   If you installed Python with conda, we recommend to create a dedicated
   conda environment with all the build dependencies of f3dasm:

   .. code-block:: console

     conda create -n mfpml_env python=3.10
     conda activate mfpml_env

3. If you run the development version, it is annoying to reinstall the package each time you update the sources.
   Therefore it is recommended that you install the package from a local source, allowing you to edit the code in-place. 
   This builds the extension in place and creates a link to the development directory (see `the pip docs <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_).

   .. code-block:: console

     pip install --verbose --no-build-isolation --editable .

4. In order to check your installation you can use

  .. code-block:: console

     $ python -c "import mfpml"
     >>> 2023-07-05 14:56:40,015 - Imported mfpml (version: 1.x.x)


5. Install the development requirements:

   .. code-block:: console

     pip install -r requirements.txt