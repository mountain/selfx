# selfx: a self explorer

The nature of self is a hot topic in human history, rich but debating. Reinforcement learning sheds new light on this old area. In this project, we investigate the potential of the reflective two-world structure:
* the outer world: the real game environment
* the inner world: a virtual world set up by the agent

Example game
=============

The code in ``selfx_billard`` is a reference example. In this example, the agent ``monster``(the yellow dot)
is like a plankton living in water swarmed by small algae (the green dots), and the ``obstacles``(the red dots)
are also part of the enviroment.

Both idle and swimming will cost energy, the only way to survive is to eat the green dots to charge energy.

The longger the monster live, the higher the game score.

In the output video, the screen is divided into upper part and bottom part. The upper part is the inner world,
while the bottom part is the outer world.


Setup and training
===================
This project follow the standard of [gym](https://gym.openai.com/) proposed by OpenAI.

Setup the enviroment
--------------------

```bash
git clone https://github.com/mountain/selfx.git
cd selfx
. hello
```

Testing the gym env
--------------------
Assuming the current directory is in the root of ``selfx``

```bash
python -m main
```

Trainning the program
--------------------
Assuming the current directory is in the root of ``selfx``

```bash
python -m mainq -g 0 -n 1000
```


