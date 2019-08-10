# S3++ 2019 ML Project


## Running

```
python3 -m pip install -r requirements.txt
python app.py 
```

Or with docker:

```
docker build . -t app
docker run --rm -it -p 8080:80 app
```


#Summary

## Puzzles

- Factorize 245143486106224403217900081922883253630
- 100000th smallest number of form `2^i * 3^j * 5 ^ k` (where `i`, `j`, `k` are nonnegative integers)
- The Josephus Problem, 41 people, where to sit?
- In how many ways can you write 100 parenthesis?
- Prove that in any 6 people there exists 3 who either are all friends or all enemies. 

## Python

- [Basic Python Tutorial](http://cs231n.github.io/python-numpy-tutorial/#python) ✅

## Linear Algebra

- [Numpy Tutorial](http://cs231n.github.io/python-numpy-tutorial/#numpy) ✅
- [Numpy Broadcasting Tutorial](http://cs231n.github.io/python-numpy-tutorial/#numpy-broadcasting) ✅

## Classic Computer Vision

- [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)

## Graph Theory

- [A Gentle Introduction To Graph Theory](https://medium.com/basecs/a-gentle-introduction-to-graph-theory-77969829ead8) ✅

## Algorithms & optimizations

- [DFS](https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/) ✅
- [BFS](https://www.geeksforgeeks.org/level-order-tree-traversal/) ✅
- [Genetic Algorithm](https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9) ✅

### Convolutional Neural Networks for Visual Recognition

- [Image Classificaiton](http://cs231n.github.io/classification/)
- [Linear classification](http://cs231n.github.io/linear-classify/)
- [Optimization](http://cs231n.github.io/optimization-1/)
- [Neural Networks Architecture](http://cs231n.github.io/neural-networks-1/)
- [Neural Networks Data and Loss](http://cs231n.github.io/neural-networks-2/)
- [Neural Networks Case Study](http://cs231n.github.io/neural-networks-case-study/)
- [Convolutional Neural Networks Cheat Sheet](http://cs231n.github.io/convolutional-networks/)
- [Convolutional Neural Networks Understanding and Visualizing](http://cs231n.github.io/understanding-cnn/)
- [Convolutional Neural Networks Transfer Learning and Fine-tuning](http://cs231n.github.io/transfer-learning/)


### MNIST Classification

- [Keras Implementation](https://keras.io/examples/mnist_cnn/)
- [Tensorflow Keras Implementation](https://www.tensorflow.org/beta/tutorials/keras/basic_classification)
- [PyTorch Implementation](https://github.com/pytorch/examples/blob/master/mnist/main.py)

# Project

- Constructing Extraction & Solving Models
- Evaluation

## Extraction

- Constructing OCR & Layouting Models
- Evaluation

### OCR

- Constructing Localization & Classification Models
- Evaluation

#### Localization

- Conversion to grayscale
- Thresholding (binarization)
- Morphological Transformations
- Finding Connected Components
- Postprocessing
- Evaluation

#### Classificaiton

- Defining Supported Characters
- Data Preparation
- Learning Classification Model
- Evaluation

### Layouting

- Preprocessing
- Sorting
- Parsing
- Evaluation

## Solving

- SimPy or own?
- Evaluation