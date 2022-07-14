import warnings

warnings.filterwarnings("ignore")

import metalgpy as mpy

# A Expression Tree can be built
a = mpy.Float(0, 10, name="a")
b = mpy.Float(0, 10, name="b")
program = a + b
print("Program: ", program, end="\n\n")

# the choice method returns the variable symbols of the symbolized program
choices = program.choices()
print("Variable Space: ", choices)

# mpy.sample(n, program) generates clones of the symbolized program
# optimize the program
for i, eval in mpy.sample(program, size=20):

    sample_program = program.clone().freeze(eval.x)
    y = sample_program.evaluate()
    print(f"{i:02d} -> {sample_program} = {y}")

    eval.report(y)
