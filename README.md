# Meta Algorithm in Python (MetalgPy)

What if we could write a program that generates programs? Inspired by Automated Machine Learning research.

> :warning: **Experimental**: Contributions are welcome!

## Install

```console
pip install metalgpy
```

## Example

A simple but detailed example:

```python
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
```

gives the following output:

```console
Program:  f(Float(id=0, low=0, high=10))

Variable Space:  {'0': Float(id=0, low=0, high=10)}

01 -> f(1.4883186068135734) = 2.215092275387496
02 -> f(9.731099196329486) = 94.69429156880437
03 -> f(2.8835900819936366) = 8.315091760972068
04 -> f(6.684879549955022) = 44.68761459740686
05 -> f(6.369117254195896) = 40.56565459769586
06 -> f(8.311599275340795) = 69.08268251384563
07 -> f(3.9495544683795036) = 15.598980498696504
08 -> f(2.719439725535402) = 7.395352420820062
09 -> f(5.076587322264285) = 25.771738840574468
10 -> f(6.509647409342488) = 42.37550939395937
11 -> f(0.0) = 0.0
12 -> f(0.07807885269930037) = 0.006096307238839045
13 -> f(0.004326455132792617) = 1.8718214016067583e-05
14 -> f(0.03447243207111301) = 0.0011883485728975008
15 -> f(0.018114237444373238) = 0.0003281255981911335
16 -> f(0.0020049360585783216) = 4.019768598987575e-06
17 -> f(0.6959012004878518) = 0.48427848084043323
18 -> f(0.006902913600794758) = 4.765021618003725e-05
19 -> f(0.04812048929037971) = 0.0023155814895455483
20 -> f(0.015496977861506611) = 0.00024015632284002603
21 -> f(0.03973738943234384) = 0.0015790601188977519
22 -> f(0.14944732113771342) = 0.022334501795238843
23 -> f(0.1538705525239814) = 0.02367614693403532
24 -> f(0.02714364492250043) = 0.0007367774596787835
25 -> f(0.013367771420287281) = 0.00017869731274504944
26 -> f(0.07504851702564763) = 0.0056322799077489225
27 -> f(0.061488350499158975) = 0.0037808172471074236
28 -> f(0.010089082470558145) = 0.00010178958509772366
29 -> f(0.1785305521706959) = 0.031873158058373575
30 -> f(0.07218850699949093) = 0.005211180542815551
31 -> f(0.08273460533704255) = 0.006845014920276189
32 -> f(0.004884441886340666) = 2.3857772541039158e-05
```
