from numpy.random.mtrand import sample
import metalgpy as mpy

# A Expression Tree can be built
a = mpy.Int(0, 10, name="a")
b = mpy.Int(0, 10, name="b")
program = a + b
print("Program: ", program, end="\n\n")

# the choice method returns the variable symbols of the symbolized program
choices = program.choices()
print("Variable Space: ", choices)

# mpy.sample(n, program) generates clones of the symbolized program
for _, sample_program in mpy.sample(program, size=5):

    print("\n ** new random program **")
    print(f"{sample_program} = {sample_program.evaluate()}")


