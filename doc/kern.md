
# The geisten kern (core) functions - version _0.6-0_


              ╚══╗ ║ ╔════╝
                 ╠═╩═╩╗
                 ║ ▒▒ ║
                 ╠═╦═╦╝
              ╔══╝ ╚═╚════╗

 > By Germar Schlegel - 2021

The geisten neural net (_gstnn_) kern module contains functions to run and train deep learning
neural networks.



 ## Macros


### ARRAY_LENGTH - Return the length (number of elements) of the c array.

#### Parameters

 - `arr` The c array

 ## Functions


### trans_f32()

Transform input data array to output array via a weight matrix.

#### Parameters

 - `batch_len` The number of parallel processed input data.
 - `m` The number of input (matrix) rows.
 - `n` The number of output (matrix) columns.
 - `w` The m x n weight matrix.
 - `x` The input vector of length `m`.
 - `y` The output (result) vector  of length `n`.


### train_sgd()

Trains the weight matrix by the delta output array.

The Stochastic gradient descent (**SGD**) is an iterative
method for optimizing an objective function with suitable smoothness
properties
(See [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)).

#### Parameters

 - `batch_len` The number of parallel input data.
 - `m` The number of input (matrix) rows.
 - `n` The number of output (matrix) columns.
 - `x` The input vector of length `m`.
 - `dy` The delta output (difference between output and expected output) vector  of length `n`.
 - `rate` The leaning rate
 - `w` The m x n weight matrix.


### loss()

Calculates `dx` from the output `dy`.

#### Parameters

 - `batch_len` The number of parallel input data.
 - `m` The number of input (matrix) rows.
 - `n` The number of output (matrix) columns.
 - `w` The m x n weight matrix.
- `dy` The delta output (difference between output and expected output)
vector  of length `n`.
- `dx` The calculated delta input vector  of length `m`.


### vec_delta()

Calculates the delta between `v1` and `v2`.
The result is written into the vector `d`

#### Parameters

 - `size` The length of the vectors.
 - `v1` The first vector.
 - `v2` The second vector.
 - `d` The calculated delta of v1-v2.
 Returns the error of the delta vector (`d`)


### vec_is_equal_f32()

Check if two vectors are equal within a tolerance range.

#### Parameters

 - `n` The length of the vectors.
 - `a` The first vector.
 - `b` The second vector.
 - `epsilon` The allowed tolerance.
 Returns true if the both vectors are equal.


### weights_norm_init()

Initialize a weight matrix with random noise values.

#### Parameters

 - `m` The rows of the weight matrix.
 - `n` The columns of the weight matrix.
 - `w` The weight matrix.

 ## Activation functions and its derivatives


### relu()

Rectified Linear Units activation function. *
The formula is: max(0,z).
It’s not linear and provides the same benefits as Sigmoid but with better performance.

#### Parameters

 - `len` The length of the vector.
 - `y` The vector.


### tanhg()

Tanh activation function.

Tanh squashes a real-valued number to the range [-1, 1]. It’s non-linear.
But unlike Sigmoid, its output is zero-centered. Therefore, in practice
the tanh non-linearity is always preferred to the sigmoid nonlinearity.

#### Parameters

 - `len` The length of the vector.
 - `y` The vector.


### tanhg()

Sigmoid activation function.

Sigmoid takes a real value as input and outputs another value between 0 and 1.
It’s non-linear, continuously differentiable, monotonic, and has a fixed output range.

#### Parameters

 - `len` The length of the vector.
 - `y` The vector.


### weights_create_or_load()

Create or load the matrix fom a memory mapped file or direct from memory.

If filename == NULL, the matrix will be allocated from memory. The weights matrix will be lost when closing the program
If the filename != NULL, the matrix will be saved and updated in the file

#### Parameters

 - `filename` The file name to save the weights matrix.
 - `input_len` The rows of the weight matrix.
 - `input_len` The columns of the weight matrix.

 Returns the allocated matrix memory.


### dropout()

Set random elements in the vector `vec` to _0_. It is allowed for `vec` and `result` to be identical (the same array).

#### Parameters

 - `len` The length of the vector.
 - `vec` The original, input vector.
 - `p` The related probability to set the element to _0_.
 - `result` The new vector with some of the elements is set to _0_.

