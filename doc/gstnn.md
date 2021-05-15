GSTNN(1) - General Commands Manual

# NAME

**gstnn** - transform the standard input to the predicted standard output

# SYNOPSIS

**gstnn**
\[**-h**]
\[**-f**]
\[**-t**&nbsp;*TARGET\_FILE*]
\[INPUT\_FILE]

# DESCRIPTION

**gstnn**
is a deep learning programm for POSIX like systems. It is running in two modes: predicting and training.
It reads input from the standard input and writes the result to the standard output.
If a FILE is given, gstnn uses the data read from the FILE as target data to fit the deep learning model.
When in training mode, the error rate and the accuracy is printed to the standard error (stderr).
Train the neural network with the target data read from
**FILE.**
The arguments are as follows:

**-f**

> Don't train (freeze) the net.

**-h**

> Print the help text.

**-t** *TARGET\_FILE*

> Set the target file to train the net.

# EXIT STATUS

The
**gstnn**
program exits 0 on success, and &lt;&gt; 0 if an error occurs.

# MONITORING

During learning, text information about the current state is sent to stderr. The comma separated output is as follows:

average error

error deviation in percent

average accuracy

error rate (percent)

average duration time of a single processing step

# EXAMPLES

Run the neural network and predict
Train the neural network

# AUTHORS

Dr. Germar Schlegel

Debian - May 15, 2021
