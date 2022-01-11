from numpy.random.mtrand import sample
import metalgpy as mpy

# the @mpy.meta decorator transform an original python code 
# into a meta-program. f is now symbolizing the original python code
@mpy.meta
def f(x):
    return x**2

# program is a symbol representing the call to f (original python code)
# where the input is a symbol representing a variable List (categorical decision variable)
program = f(mpy.List([0,1,2,3,4]))
print("Program: ", program, end="\n\n")

# the choice method returns the variable symbols of the symbolized program
choices = program.choice()
print("Variable Space: ", choices)

# mpy.sample(n, program) generates clones of the symbolized program
for _, sample_program in mpy.sample(program, size=5):

    print("\n ** new random program **")
    print(f"{sample_program} = {sample_program.evaluate()}")


