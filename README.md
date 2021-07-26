# Feature Match Algorithm
## Usage
```shell=
  ./feature_match -i <scene image> -t <test image> -r <ratio threshold> -a <matching algorithm> - k <number of clusters>               
```
- ```-i <scene image>```: the image path for scene.
- ```-t <test image>```: the image path for object.
- ```-r <ratio threshold>```: the ratio of the Lowe's ratio test.
- ```-a <matching algorithm>```: choose surf algorithm or sift algorithm.
- ```-k <number of clusters>```: set clusters number.
