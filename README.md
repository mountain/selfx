# selfx: a self explorer

The nature of self is a hot topic in human history, rich but debating. Reinforcement learning sheds new light on this old area. In this project, we investigate the potential of the reflective two-world structure:
* the outer world: the real game environment
* the inner world: a virtual world set up by the agent

Example game
=============

The code in ``selfx_billard`` is a reference example. In this example, the agent ``monster``(the yellow dot)
is like a plankton living in water swarmed by small algae (the green dots), and the ``obstacles``(the red disc)
are also part of the enviroment.

Both the action of idle and swimming will cost energy, the only way for monster to survive is to eat algae to charge energy.

The longer the monster live, the higher the game score is.

In the output video, the screen is divided into three parts:
* the local view from the point of the monster
* the global view of the inner world
* the global view of the outer world

Only the first two views - the local view and the global view of the inner world - are accessible by the monster and
uses them as inputs of neural network.

<img src="https://raw.githubusercontent.com/mountain/selfx/master/docs/demo.png" alt="demo" width="460px" height="690px" style="margin: 100px">

After training for about 100 episode, the monster was learned to avoid obstacles, but it could merely draw randomly on the inner world.

[<img src="https://img.youtube.com/vi/Sh2LnCXzE8A/maxresdefault.jpg" width="50%">](https://youtu.be/Sh2LnCXzE8A)


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


