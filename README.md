# Under Construction


[Project page](https://haato-w.github.io/sketch-rod-gs-project-page/) | [Paper]() | [Video](https://youtu.be/eaK0p0nU47g?si=sTGmfLNSeCYiJELJ) | [Dataset (Google Drive)](https://drive.google.com/drive/folders/1QhOkshES3-ubzQtoMD1wOpd_6Vj45H0f?usp=sharing) <br>

![Teaser image](assets/teaser.jpg)


This repo contains the official implementation for the paper "SketchRodGS: Sketch-based Extraction of Slender Geometries for Animating Gaussian Splatting Scenes"(SIGGRAPH Aisa 2025 Technical Communications). Our work presents a method to extract a polyline representation of slender part of the objects in Gaussian splatting scene from the user's sketching input. Our method robustly construct a polyline mesh that represents the slender parts using the screen-space shortest path analysis that can be efficiently solved dynamic programming.

# How to use
Please download GS .ply data into `gs_data` dir.<br>
`git clone https://github.com/haato-w/gs-string.git --recursive`<br>
`conda env create --file environment.yml`<br>
`conda activate gs-string`<br>
`python viewer.py`<br>

# Installation

