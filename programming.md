# Python Notes

# Broadcasting
    A technique in python that is similar to storing matrix data in a sparse
    matrix in that it allows multiplying a 1d matrix with a 2d one without
    modifying its shape.
    It allows massive speed improvements over simple python looping if properly
    used.

# Class vs Dict
    Storing grouped data into a dict will always be faster, but if there is a
    possibility of additional logic that will be added to the structure then
    a class is the way to go.
    Class `__slot__` is a new method that speeds up read/writes from a class
    instance.
    [link](https://stackoverflow.com/questions/4045161/should-i-use-a-class-or-dictionary)

# Writing Tests
    Tests should be used to test all functionality of the data layer. ?? (where
    data is stored/manipuated) but not for connectors/interfaces. Rule of thumb
    is to build tests as before and as you modify your functions/classes. If tests
    are too hard to write, its likely the design is suboptimal/too complex.

#
