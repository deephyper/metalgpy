import warnings
warnings.filterwarnings("ignore")

import metalgpy as mpy

# the @mpy.meta decorator transform an original python code 
# into a meta-program. f is now symbolizing the original python code
@mpy.meta
def f(x):
    return x**2

# program is a symbol representing the call to f (original python code)
# where the input is a symbol representing a variable List (categorical decision variable)
program = f(mpy.Float(0, 10))
print("Program: ", program, end="\n\n")

# the choice method returns the variable symbols of the symbolized program
choices = program.choices()
print("Variable Space: ", choices)

# optimize the program
for i, eval in mpy.sample(program, size=100):

    sample_program = program.clone().freeze(eval.x)
    y = sample_program.evaluate()
    print(f"{i:02d} -> {sample_program} = {y}")

    eval.report(y)


