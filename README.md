# selfx: a self explorer

The nature of self is a hot topic in human history, rich but debating. Reinforcement learning sheds new light on this old area. In this project, we investigate the potential of the reflective two-world structure:
* the outer world: the real game environment
* the inner world: a virtual world set up by the agent

Setup and training
======
This project follow the standard of [gym](https://gym.openai.com/) proposed by OpenAI.

##Setup the enviroment

```bash
cd selfx
. hello
```

##Testing the gym env

```bash
cd selfx
python -m main
```

##Trainning the program

```bash
cd selfx
python -m mainq -g 0 -n 1000
```


