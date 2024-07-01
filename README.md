# Intro

Developed by VITO Remote Sensing, CORSA revolutionizes data compression and processing for Earth Observation.
Using advanced deep learning, it efficiently compresses imagery from sensors like Sentinel-1, Sentinel-2 and PRISMA, drastically reducing data volume while maintaining high image fidelity, even at 100x compression rates.

CORSA's compressed features can be directly used to build downstream applications like land use classification change detection and natural disaster mapping and others.

read more about CORSA:
- VITO blog: https://blog.vito.be/remotesensing/corsa
- Services: https://remotesensing.vito.be/services/corsa

<br>
<img src="images/corsa_flow.PNG" alt="The CORSA processing flow" width="800"/>

# Overview
In this repository we provide a demo for CORSA applied on Sentinel-2 data, implemented on the Terrascope platform.
The CORSA workflow compresses the 10m and 20m resolution bands of Sentinel-2 (10 bands) to compact tif files.
In the notebook we:
- Compress large regions of Europe using the CORSA model
- Uncompress (part of this) region to showcase the near-lossless compression
- Use the compressed features in a downstream landcover classification task


# Get started
0. If you don't have a Terrascope account already create follow the authentication link at the beginning of the notebook or create an account for free here: https://terrascope.be/

1. To create the right environment the easiest option is to use conda ```conda env create -f environment.yml``` <br>
Otherwise, it's still possible to install all the dependencies listed in the yml file with pip.

2. Dive into the self-explanatory notebook 'corsa_explore.ipynb'