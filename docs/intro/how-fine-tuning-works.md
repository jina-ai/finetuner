(technical-details)=
# {octicon}`mortar-board` How Fine-tuning Works


<!-- Here should be the guides from [Part 1](https://www.notion.so/Not-So-Brief-Explanation-of-Neural-Networks-dd8cd521023642f792683ff43bd2ccf1), 
[Part 2](https://www.notion.so/Transfer-learning-and-fine-tuning-d405c1fee3d4444cb343a05e73f02db6), ... -->

# Fine-tuning

To understand fine-tuning, you will need to understand how neural network models work, 
in general, if not in fine detail.

This document is intended for readers with some understanding of algebra and 
computer engineering but who may not have any experience with machine learning 
or artificial intelligence. If any section covers subjects you are already 
familiar with, feel free to skip it.

We also have a [non-technical summary of what fine-tuning does](../what-is-finetuner/), 
but without explaining how it works.

## What are Neural Networks?

Neural Networks are a family of computational techniques for data analysis 
and machine learning that underlie most of the recent breakthroughs in 
artificial intelligence. Although neural networks have a history dating 
back to the 1940s, they have come to almost completely dominate AI 
research in the last decade due to a collection of breakthroughs in 
scaling and training techniques that were originally called *Deep Learning*.

Some people insist on calling them *artificial neural networks*, to distinguish 
them from the nervous systems of animals and people. Although they were 
originally inspired by biology, neural network technology has progressed 
entirely independently of advances in neurology, and we know today that 
biological cognition does not work the way artificial neural networks do. The 
term “neural network” itself is almost exclusively used by computer 
professionals and rarely if ever by neurologists, so the addition of 
“artificial” is superfluous.

We also sometimes call them “neural models”, “machine learning models”, 
“AI models” or when the context is clear enough, just “models”. At present, 
these terms are largely interchangeable.

Neural networks have some important properties that make them central to recent 
work in machine learning:

1. Neural networks can, in principle, perform almost any kind of data 
transformation, if they have the right configuration and are large enough. 
Any problem that requires consistent mapping from digital inputs to digital 
outputs can potentially be solved by a neural network.
2. Neural networks are highly scalable and parallelizable. They scale 
linearly both in time and in computations. If you want your model to run in 
half as much time, you can just devote twice as many computer cores to it.
3. Neural networks can learn to generalize from examples and sometimes find 
good solutions to problems where even humans don’t have a good solution. 
They can genuinely learn from the data presented to them, sometimes better 
than humans can, but not in the same way that humans do.

To explain how this works, we need to use a bit of math.

## Vectors

Neural networks work using a common abstraction: Many real-world problems can 
be recast as vector transformation problems, and a neural network is, in its 
most generic form, a scheme for transforming vectors.

Vectors are ordered lists of numbers that correspond to points in a high-dimensional 
metric space. This means that:

- A vector corresponds to a single, unique point, and each point corresponds to a 
unique vector.
- If two vectors are the same, the points they correspond to are the same.
- There are functions called *metrics* or *distance functions* that measure the 
distance between any two vectors. If two vectors are the same, the distance 
between them is zero. If two vectors are not the same, the distance is some 
amount greater than zero.

Please read our brief refresher on vectors if you are not already very familiar 
with the mathematical concept: [Brief Refresher on Vectors](../brief-refresher-on-vectors/)


> Computer data, like a vector, is just a sequence of numbers. So any digital 
> information can be treated like a vector just by calling it one!

If we look at it that way, any problem that involves taking some finite amount 
of computer data as input and turning it into some other computer data as output, 
is a problem that involves mapping vectors in some vector space into other 
vectors in another (or possibly the same!) vector space.  This makes neural 
networks a very general-purpose technique for doing things. It is not necessarily 
the best way to address all problems, but it is a very effective way to address 
a great many problems.

In practice, we typically do transform the input data in some way before inputting 
it into a neural network (we call this *preprocessing*) and have to perform some 
transformation on the output. Nonetheless, in theory, all data transformation 
problems can be solved by some neural network. The only fundamental requirement 
for a neural network solution is that we be able to express the problem we want 
to solve as a mapping from some data to some other data.

There are many problems that are impractical to use neural networks to solve, 
even if in theory we could solve them that way. Traditional kinds of 
programming are much better suited to problems where precision and exact 
repeatability are important, or where we already know an exact procedure 
for doing a task.

For example, it would be very impractical to train a neural network to do 
arithmetic because much simpler devices are already very good at that kind of 
math, and because most people would be unhappy with a calculator that adds 2 
and 2 but is only guaranteed to answer with something between 3 and 5 most 
of the time.

But neural networks are very good at problems where we have no solution, but 
we know there is one because we do it ourselves. Image recognition, OCR and 
handwriting recognition, speech recognition, and natural language translation, 
to name a few things, have proven impossible to reduce to a simple procedure, 
but we know they can be done because we pay people to do them.

These are the tasks where neural networks excel. Instead of programming 
computers to do those tasks, we just present them with examples, and 
they program themselves.

## How Neural Networks Work

The image below is a very schematized picture of a small neural network:

![Small neural network](../imgs/nn_small.png)

This particular network maps vectors with three values `[x, y, z]` to vectors 
with two values `[a, b]`. It has two “hidden” layers, which are also vectors, 
in this case, each has four values. Neural networks can have any size and in 
principle map vectors of any size to other vectors of any size, using any number 
or configuration of hidden layers, but we are going to use the one pictured 
above for this example.

The way neural networks work is that we take the input vector, in this case, 
we’ll use  `[x, y, z]`, and multiply it by a matrix of numbers called 
“weights” to get the values of the first hidden layer.

In linear algebra notation, it looks like this:

![Single-layer matrix equations](../imgs/matrix1.png)

Although all these equations look formidable, they are really just multiplying 
a bunch of numbers and then adding them up.

In the same way, the second hidden layer gets its values by applying a matrix 
of weights to the values of the first hidden layer:

![Second layer matrix equations](../imgs/matrix2.png)

And then to calculate the output layer, we multiply the last hidden layer by 
another matrix of weights:

![Last hidden layer matrix equations](../imgs/matrix3.png)

Most neural networks then apply what is called an *activation function* to 
the values of each hidden layer before using it to calculate the next layer. 
The activation function is usually a threshold or a function that acts like 
a threshold, usually a [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function). 
There are also some other things that people sometimes do between hidden layers. 
Although essential to neural network engineering and design, we don’t need to 
discuss these here to make sense of fine-tuning.

The essential part is to understand how input vectors are transformed by 
multiplying them by weight matrices and then summing, over and over, until 
you’ve produced a final output vector.

## Training Neural Networks

When training a network, training data consists of pairs of input vectors `[x, y, z]` 
and output vectors `[a, b]`. The training process itself follows a simple scheme:

1. Enter into the neural network each input vector from the training data`[x, y, z]`.
2. Calculate the resulting output vector `[a', b']`
3. Calculate the distance between the output vector `[a', b']` and the correct 
value `[a, b]` from the training data, using some distance metric, usually Euclidean 
distance or cosine distance (described in the [Brief Refresher on Vectors](https://www.notion.so/Brief-Refresher-on-Vectors-128bfb422b574f4b8212694154ee2061)).
4. Adjust the weights in the neural network just a little bit, proportionate to 
the distance calculated in the previous step, so that the next time, the output vector
`[a', b']` will be a little bit closer to the correct value `[a, b]`.

This is done over and over and over, many times, with different example input-output 
pairs, slowly adjusting the weights to make the neural network model produce output 
vectors closer and closer to the correct ones. There are a number of procedures for 
calculating how to adjust the weights, but the traditional way is called 
*back-propagation*, and it’s an application of a common method in statistics called 
*regression*.

There are some additional techniques used in training to enhance robustness and 
speed up processing, and there are other algorithms for adjusting the weights 
that are sometimes used in AI model training. Nonetheless, the purpose is 
always the same: To induce the weights to converge on values that optimally 
produce the expected outputs. Over time, the network learns to translate input 
vectors into output vectors in a way that should generalize to new data and to 
do a good job of the task expected of it.

This example neural network is very small. It has only eight nodes in the hidden 
layers, and a total of 36 weights (3 x 4 + 4 x 4 + 4 x 2 = 36). By comparison, a 
general-purpose vision processing model like ViT-L/16 takes as input color images 
with a resolution of 224x224 pixels. This means it takes an input vector of 
approximately 150,000 values. It uses 24 hidden layers, has 307 million weights, 
and outputs a vector with 16 values. State-of-the-art AI models are vastly larger 
than the example above, but the principles behind how they learn remain the same.

## Example: Recognizing pictures of fruit

Let’s consider a more concrete example. Imagine we trained a neural network to 
take as input pictures of fruit, and then tell us which fruit it is. To make 
the problem simpler, let’s say it can recognize different kinds of fruit, 
specified in advance.

We don’t need to be very specific about the exact number of nodes and hidden 
layers, the network will look basically like this:

![Fruit neural network](../imgs/fruit_NN2.png)

The last hidden layer will be a vector with some high number of dimensions. 
Since it’s very hard to draw that, we’re going to pretend the last hidden layer 
forms a two-dimensional space. Before training, when all the weights are set to 
random values, we would expect that layer to give us randomly distributed 
results when we enter pictures of fruit:

![Fruit embeddings pre-training](../imgs/scatterplot_fruit.png)

After training, we would expect the last hidden layer’s distribution to look 
more like this:

![Fruit embeddings post-training](../imgs/scatterplot_fruit_ordered.png)

In short, the last hidden layer should be organized so that the output layer 
can easily recognize fruits just by drawing lines around the parts of the 
vector space each fruit falls into.

![Fruit embeddings partitioned](../imgs/scatterplot_fruit_partitioned.png)

Then, when we present the model with new images, we can expect them to appear 
in the right part of the vector space so that they are easily recognized:

![New fruit inputs](../imgs/fruit_and_dots.png)

## Embeddings

In classification problems, we typically have an explicit output layer, like 
in the fruit-recognition example above. We provide one output node for each 
class we want to recognize. Whenever we want to classify inputs into some 
well-defined set of classes, we can construct this kind of network: We transform 
input vectors into output vectors where each dimension is linked to a specific 
output class.

All data classification problems are vector transformation problems, however, 
not all vector transformation problems are classification problems.

For example, consider the problem of face recognition. Let’s say you have a 
database of some large number of photos of people’s faces, and, when given a 
new picture of a face, you want to retrieve the most similar face from the 
database.

Let’s say there are a million faces in the database. You could try to treat 
this as a classification problem: Create a neural network with a million 
output nodes — one for each face — and then train it to class new pictures 
into one of those million categories.

This is possible, in principle, but in practice it doesn’t work very well.

Neural networks learn by adjusting their weights in response to examples. To 
learn to do classification, they need to see many examples of each class that 
they have to learn. How many depends on factors like the size of the network, 
the quality of the data, and the difficulty of the problem. It is very difficult 
to know in advance how much data is enough, which is why training data sets are 
typically as large as possible.

If treat each of a million unique faces as a discrete category, we might need 
hundreds, thousands, or even more pictures of each face to learn to classify 
new pictures correctly. This is very difficult to do, and very inefficient.

There is a better way to look at the problem: Instead of treating it like a 
classification problem, we can treat it like an information retrieval problem. 
Instead of asking what category each picture belongs to, we ask: Given an input 
picture, what are the most similar pictures to it in our database, and are any 
of them similar enough to be a good match?

We saw with the fruit classification problem how the last hidden layer becomes 
organized during training so that the same kinds of fruits cluster together in 
the vector space defined by that layer. When we train a neural network so that 
it maps input data to a vector space where the placement in that space encodes 
useful information about the input data, we call vectors in that space *embeddings*.

In the fruit recognition example above, the last hidden layer of the network 
produces embeddings, because we can see how the locations of the output 
vectors encodes information about what kind of fruit is in the input image.

We can apply that principle to face recognition.

What we would like from a face recognition neural model is for it to transform 
pictures of faces as inputs into embeddings, such that the more two pictures of 
faces are similar, the more their embeddings will be close together. The idea 
is that if two pictures are of the same person, their embeddings will be very 
close together, much closer than two pictures of different people.

Embeddings are high-dimensional vectors in all practical cases, but once again, 
since it’s very hard to draw high-dimensional spaces, we’ll pretend its just 
two dimensions for the example below:

![Face embeddings](../imgs/Faces.png)

We can see in this example that some superficial features of faces cluster 
together. For example, the men and women are separated:

![Face embeddings by gender](../imgs/MenWomen.png)

And people are clustered by features of their hair:

![Face embeddingsby hair type](../imgs/HairType.png)

In real use cases, embeddings have a lot of dimensions and form a very 
high-dimensional vector space. With so many dimensions, different features can 
cluster together in ways that we can’t really draw in two dimensions.

There are a number of techniques for constructing and training networks like 
this, but what’s important to understand is what these kinds of models do and 
how they work.

In this example, we want the model to produce an embedding for each picture in 
the database, which we then store. Then, we take a new picture, we get its 
embedding vector from the model, and then we compare that vector to the stored 
vectors in the database. The result will be a ranking of all the images in the 
database by how similar they are to the new picture.

If we’ve trained the network correctly, and there is another picture of the same 
person in the database, the closest embedding from the database will match a 
picture of the same person.

![Wynona Ryder as embedding](../imgs/Wynona.png)

This reduces the problem of face recognition to identifying the stored image 
whose embedding is closest to the embedding of the query image, and then deciding 
if they are close enough to be the same person.

This same logic is used for many other problems, like multimodal information 
retrieval. For example, if we have a database of pictures, and we just want 
to retrieve pictures of dogs. We solve this problem by constructing and 
co-training two models, one that takes images as input, and one that takes 
text as input, but both output embeddings in the same vector space.

![Dog embeddings](../imgs/dog_w_text.png)