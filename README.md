# VGAE_paper

This repo contains my implementation and extension on the paper Variational Graph Auto-Encoders by Thomas Kipf and Max Welling for the course Machine Learning for Graphs (VU Amsterdam).

Original paper: https://dblp.org/rec/journals/corr/KipfW16a.html

By editing the parameters at the end each file, and running it, you can reproduce any result.

All .py file contain standalone implementations of the code from loading data to training loops:
- *original.py* contains the reimplementation of the original models
- *extension.py* contains the implementation of my extension
- *original_newdata.py* contains the code for running the original models on the new directional graphs used for the extension as a baseline