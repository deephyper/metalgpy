# Meta Algorithm in Python (MetalgPy)

What if we could write a program that generates programs? Inspired by Automated Machine Learning research such as [PyGlove](https://proceedings.neurips.cc/paper/2020/file/012a91467f210472fab4e11359bbfef6-Paper.pdf).

> :warning: **Experimental**: Contributions are welcome!

## Install

```console
pip install metalgpy
```

## Example

A simple but detailed example:

```python
import metalgpy as mpy

# the @mpy.meta decorator transform an original python code 
# into a meta-program. f is now symbolizing the original python code
@mpy.meta
def f(x):
    return x

# program is a symbol representing the call to f (original python code)
# where the input is a symbol representing a variable List (categorical decision variable)
program = f(mpy.List([0,1,2,3,4]))
print("Program: ", program, end="\n\n")

# the choice method returns the variable symbols of the symbolized program
choices = program.choice()
print("Variable Space: ", choices)

# mpy.sample(n, program) generates clones of the symbolized program
for sample_program in mpy.sample(5, program):

    print("\n ** new program **")

    # we iterate over all variables of the variable space and randomly sample each of them
    choice = [v.sample() for v in choices]
    print("choice: ", choice)

    # we freeze the sampled program with a choice for each variable
    sample_program.freeze(choice)
    print("frozen program: ", sample_program)

    # we can now evaluate the program
    res = sample_program.evaluate()
    print("evaluation: ", res)
```

gives the following output:

```console
Program:  f(List(0, 1, 2, 3, 4))

Variable Space:  [List(0, 1, 2, 3, 4)]

 ** new program **
choice:  [3]
frozen program:  f(3)
evaluation:  3

 ** new program **
choice:  [4]
frozen program:  f(4)
evaluation:  4

 ** new program **
choice:  [3]
frozen program:  f(3)
evaluation:  3

 ** new program **
choice:  [2]
frozen program:  f(2)
evaluation:  2

 ** new program **
choice:  [4]
frozen program:  f(4)
evaluation:  4
```
