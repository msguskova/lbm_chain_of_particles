# title :)


## Chain of particles 
```python diff_rads_chain_shifted.py --help
usage: diff_rads_chain_shifted [-h] [-i [INTERPARTICLE_DISTANCE]] W cs2 u_flow nu N cy0 rad2

Chain of rigid particles flows in Poiseuille. Particles of the equally distributed over length of the channel.
Particles of two radii are alternating in the chain Interparticle distance can be set, the default value is 4R.

positional arguments:
  W                     width of the channel in number of nodes (41 will give w = 40)
  cs2                   sound speed of the model
  u_flow                max velocity for poiseuille flow
  nu                    kinematic viscosity
  N                     number of particles in the chain
  cy0                   y-coordinate for initial starting position
  rad2                  Radius of bigger particle

optional arguments:
  -h, --help            show this help message and exit
  -i [INTERPARTICLE_DISTANCE], --interparticle_distance [INTERPARTICLE_DISTANCE]
                        Distance between particles centers measured in radius of the particles```

## Array of cylinders

```python array_of_cylinders.py --help
usage: array_of_cylinders [-h] W cs2 u_flow nu

One rigid particle flows in Poiseuille.

positional arguments:
  W           width of the channel in number of nodes (41 will give w = 40)
  cs2         sound speed of the model
  u_flow      max velocity for poiseuille flow
  nu          kinematic viscosity

optional arguments:
  -h, --help  show this help message and exit```