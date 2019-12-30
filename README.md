# selfx: a self explorer

The nature of self is a hot topic in human history, rich but debating. Reinforcement learning sheds new light on this old area. In this project, we investigate the potential of the reflective two-world structure:
* the outer world: the real game environment
* the inner world: a virtual world set up by the agent

Example game
=============

The code in ``selfx_billard`` is a reference example. In this example, the agent ``monster``(the yellow dot)
is like a plankton living in water swarmed by small algae (the green dots), and the ``obstacles``(the red dots)
are also part of the enviroment.

Both the action of idle and swimming will cost energy, the only way for monster to survive is to eat algae to charge energy.

The longer the monster live, the higher the game score is.

In the output video, the screen is divided into two parts. The upper part is the inner world,
while the bottom part is the outer world.

![demo][demo_img]

[demo_img]: https://raw.githubusercontent.com/mountain/selfx/master/docs/demo.png

As the above picture shows, at the beginning of training, the monster will draw randomly on the inner world.
 

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
. hello
python -m main
```

Trainning the program
--------------------
Assuming the current directory is in the root of ``selfx``

```bash
. hello
python -m mainq -g 0 -n 1000
```

Inference a model
------------------
Assuming the current directory is in the root of ``selfx``

```bash
. hello
python -m demo
```


