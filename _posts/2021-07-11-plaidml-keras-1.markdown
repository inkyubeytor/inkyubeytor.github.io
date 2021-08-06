---
layout: post
title:  "Using PlaidML with Keras, Part 1: Installation and Tensors"
categories: [Deep Learning]
tags: Keras PlaidML
---

Given that my primary (desktop) computer for the past few years has always had
an AMD graphics card, I was reliant on laptops and cloud solutions for
running intensive deep learning tasks.
Now that the field has somewhat matured, I took another attempt at using this
GPU for deep learning.

While PyTorch for AMD ROCm exists and supports most new AMD GPUs, I wanted to
work with Keras, so I chose to forego this option for now.
I also decided to attempt this process using only my native operating system of
Windows, rather than working with WSL or virtual machines.

In order to provide a comparison point for the benchmarks provided in this 
post, I included a partial list of system specifications below:
 * GPU: AMD RX 580 (VRAM: 8GB GDDR5)
 * CPU: Intel i7-6700k
 * RAM: 16GB DDR4-2133
 * Operating System: Windows 10 Pro

# Why PlaidML?

Keras is a deep learning API designed to provide programmers with a consistent
and easy to use interface across a variety of supporting low-level software
backends.
As the current Keras is tightly interwoven with TensorFlow, I first looked for
an adapted TensorFlow backend that works on AMD devices.

The default TensorFlow backend is built on top of the Nvidia-specific CUDA.
Adaptations look to adapt TensorFlow to OpenCL, which works with AMD GPUs
as well.
However, the two most prominent such projects I could find, 
[tensorflow-opencl](https://github.com/benoitsteiner/tensorflow-opencl) and
[tf-coriander](https://github.com/hughperkins/tf-coriander),
both appear to have been somewhat abandoned.

I chose to go with an alternative Keras backend, 
[PlaidML](https://github.com/plaidml/plaidml), which has more stars on
the repository and has been updated more recently.

# Installation

I started by creating a new PyCharm project and virtual environment with Python
3.9 and running the suggested `pip install plaidml-keras plaidbench`, which
installed PlaidML 0.7.0, Keras 2.2.4, and NumPy 1.21.0, among other packages.
I then ran `plaidml-setup` with `opencl_amd_ellesmere.0` as my chosen device.
At this point some tutorials meant for Apple devices suggest selecting the
Apple Metal listing for the GPU over the OpenCL for greater performance, but
for my personal setup this decision did not apply and `opencl_amd_ellesmere.0`
was my only non-CPU device.

The next suggested step was to run `plaidbench keras mobilenet`.
However, this resulted in the following output:

{% highlight bash %}
Running 1024 examples with mobilenet, batch size 1, on backend plaid
INFO:plaidml:Opening device "opencl_amd_ellesmere.0"
Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf.h5
17227776/17225924 [==============================] - 2s 0us/step
'str' object has no attribute 'decode'
Set --print-stacktraces to see the entire traceback
{% endhighlight %}

Printing the stack traces as suggested revealed the following lines

{% highlight bash %}
File "d:\pycharmprojects\keras-plaidml-tutorial\venv\lib\site-packages\keras_applications\mobilenet.py", line 296, in MobileNet
model.load_weights(weights_path)
File "d:\pycharmprojects\keras-plaidml-tutorial\venv\lib\site-packages\keras\engine\network.py", line 1165, in load_weights
saving.load_weights_from_hdf5_group(
File "d:\pycharmprojects\keras-plaidml-tutorial\venv\lib\site-packages\keras\engine\saving.py", line 1004, in load_weights_from_hdf5_group
original_keras_version = f.attrs['keras_version'].decode('utf8')
AttributeError: 'str' object has no attribute 'decode'
{% endhighlight %}

A little of research showed that this was caused by Python 2 legacy code, 
which has a 
[str.decode](https://docs.python.org/2.7/library/stdtypes.html#str.decode)
method, while Python 3 
[does not](https://docs.python.org/3/library/stdtypes.html#str.encode).
As the error resulted from the specific model rather than a failure of 
PlaidML, I cycled through the available models for `plaidbench` until I found
one that worked.
With `plaidbench keras resnet50`, the following successful result appeared:

{% highlight bash %}
Running 1024 examples with resnet50, batch size 1, on backend plaid
INFO:plaidml:Opening device "opencl_amd_ellesmere.0"
Compiling network... Warming up... Running...
Example finished, elapsed: 5.609s (compile), 24.831s (execution)

-----------------------------------------------------------------------------------------
Network Name         Inference Latency         Time / FPS
-----------------------------------------------------------------------------------------
resnet50             24.25 ms                  5.14 ms / 194.57 fps
Correctness: PASS, max_error: 4.442002591531491e-06, max_abs_error: 4.470348358154297e-07, fail_ratio: 0.0
{% endhighlight %}

In order to have a baseline to compare this to, I reran `plaidml-setup` and
chose the `llvm_cpu.0` option.
At this point, saving settings to the `.plaidml` file is required for 
`plaidbench` to use the newly selected device.

Running the same benchmark on my CPU resulted in

{% highlight bash %}
Running 1024 examples with resnet50, batch size 1, on backend plaid
INFO:plaidml:Opening device "llvm_cpu.0"
Compiling network... Warming up... Running...
Example finished, elapsed: 11.550s (compile), 3096.754s (execution)

-----------------------------------------------------------------------------------------
Network Name         Inference Latency         Time / FPS
-----------------------------------------------------------------------------------------
resnet50             3024.17 ms                3020.33 ms / 0.33 fps
Correctness: PASS, max_error: 8.304646144097205e-06, max_abs_error: 1.2814998626708984e-06, fail_ratio: 0.0
{% endhighlight %}

showing a dramatic speedup from CPU to GPU.

# Tensors

I chose to use the
[Keras tutorial for researchers](https://keras.io/getting_started/intro_to_keras_for_researchers/)
to focus on using Keras with PlaidML and avoid external packages such as
TensorBoard.

The first step of the guide is to import TensorFlow and import Keras from that.
Adapting this to use Keras and PlaidML, I tried to import PlaidML and the
associated Keras backend.

{% highlight python %}
import plaidml.keras
from plaidml.keras import backend as K

x = K.constant([[5, 2], [1, 3]])
print(x)
{% endhighlight %}

However, this resulted in an attempt to import TensorFlow from Keras while
importing the backend.

According to PlaidML documentation, the Keras backend must be explicitly set as
`"backend": "plaidml.keras.backend"` in `~/.keras/keras.json`, or the 
equivalent `C:\Users\username\.keras\keras.json` for Windows.
An alternative is to set the environment variable 
`KERAS_BACKEND=plaidml.keras.backend`.
Trying these methods resulted in

{% highlight python %}
ValueError: Invalid backend. Missing required entry : placeholder
{% endhighlight %}

each time, so I went with the deprecated option

{% highlight python %}
import plaidml.keras
plaidml.keras.install_backend()

from plaidml.keras import backend as K

x = K.constant([[5, 2], [1, 3]])
print(x)
{% endhighlight %}

which worked as intended.
This managed to get past the issue of using the right backend, but resulted in 
a new error, for which the tail of the stack trace is below:

{% highlight python %}
multiarray.copyto(a, fill_value, casting='unsafe')
File "<__array_function__ internals>", line 5, in copyto
ValueError: could not broadcast input array from shape (2,2) into shape (2,)
{% endhighlight %}

The error mentioning that the target shape is `(2,)` implied an issue with 
reading the shape of the input at some point, perhaps due to treatment of the 
inner lists as objects.
Converting the list into a NumPy array as

{% highlight python %}
x = K.constant(np.array([[5, 2], [1, 3]]))
print(x)
{% endhighlight %}

resulted in the output

{% highlight python %}
INFO:plaidml:Opening device "opencl_amd_ellesmere.0"
constant_1 Tensor FLOAT32(2, 2)
{% endhighlight %}

as desired.
PlaidML appears to compute the target shape of the desired constant in a 
conservative way that treats the elements of tuples or lists as objects,
even if they are NumPy arrays.
It then attempts to fit the input into this shape using NumPy functions, which 
treat the lists and tuples as nested arrays, resulting in possible shape 
mismatches.

Moving on, the next 3 lines of the tutorial are:

{% highlight python %}
x.numpy()
print("dtype:", x.dtype)
print("shape:", x.shape)
{% endhighlight %}

For `plaidml.tile.Value`s, the type of values returned by a variety of PlaidML
function implementations including `K.constant`, the equivalent to the `numpy`
method is `eval`.

The PlaidML implementation of the `shape` method returns a `plaidml.tile.Shape`
rather than a tuple.
However, `plaidml.tile.Shape`s have `dtype` and `dims` fields providing both of
the required pieces of information.
My PlaidML-adapted code was

{% highlight python %}
x.eval()
print("dtype:", x.shape.dtype)
print("shape:", x.shape.dims)
{% endhighlight %}

The remainder of the "Tensors" section of the selected Keras tutorial consists
of convenience methods to create tensors.

{% highlight python %}
print(tf.ones(shape=(2, 1)))
print(tf.zeros(shape=(2, 1)))

x = tf.random.normal(shape=(2, 2), mean=0.0, stddev=1.0)

x = tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype="int32")
{% endhighlight %}

The PlaidML equivalents return `plaidml.tile.Value`s (printing the associated
tensors requires invoking the `eval` method), but otherwise appear to have
worked as intended.

{% highlight python %}
print(K.ones(shape=(2, 1)))
print(K.zeros(shape=(2, 1)))

x = K.random_normal(shape=(2, 2), mean=0.0, stddev=1.0)

x = K.random_uniform(shape=(2, 2), minval=0, maxval=10, dtype="int32")
{% endhighlight %}

# Conclusion

The next post in this series will continue converting the tutorial code into 
PlaidML-compatible code, including the sections "Variables," 
"Doing math in TensorFlow," and "Gradients" to the extent they can be done in
PlaidML.
