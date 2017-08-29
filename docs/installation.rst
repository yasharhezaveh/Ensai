============
Installation
============

You will need to have a working version of python. You'll also need the following libraries/packages:

1. tensorflow (instruction here: https://www.tensorflow.org/install/)
2. PIL
3. scipy/numpy
4. matplotlib (if you'd like to generate plots in the ipython notebook)

Once you have these, you can simply clone the repository with git::

    $ git clone git@github.com:yasharhezaveh/Ensai.git

Then, you'll need to download these data files:

1. `The trained network weights`_
2. `a sample of lensing images to demonstrate the tool`_
3. `a few cosmic ray and artifact maps`_

.. _`The trained network weights`: https://stanford.box.com/s/7wtkx1fr77156uec8h8apqm9my0aevpi
.. _`a sample of lensing images to demonstrate the tool`: https://stanford.box.com/s/tb2lpk824kee22ah3gz5b50trbp30vyx
.. _`a few cosmic ray and artifact maps`: https://stanford.box.com/s/hn6l82pkmhm65xsls6g7tcjq63blj8v7


After downloading these either double click on them in Mac, or untar them from the commandline with::

    $ tar xvfz CosmicRays.tar
    $ tar xvfz SAURON_TEST.tar
    $ tar xvfz trained_weights.tar

Make sure that the unpacked folders are inside "data/" folder in Ensai. Now you should be able to run the ipyn notebook or play with the python scripts!
