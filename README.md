# Deep Reinforcement Learning with TensorFlow 2.1

Source code accompanying the blog post
[Deep Reinforcement Learning with TensorFlow 2.1](http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/).

In the blog post, I showcase the `TensorFlow 2.1` features through the lens of deep reinforcement learning
by implementing an advantage actor-critic agent, solving the classic `CartPole-v0` environment.
While the goal is to showcase `TensorFlow 2.1`, I also provide a brief overview of the DRL methods.

You can view the code either as a [notebook](actor-critic-agent-with-tensorflow2.ipynb),
a self-contained [script](a2c.py), or execute it online with
[Google Colab](https://colab.research.google.com/drive/1XoHmGiwo2eUN-gzSVLRvE10fIf_ycO1j).

To run it locally, install the dependencies with `pip install -r requirements.txt`, and then execute `python a2c.py`.  

To control various hyperparameters, specify them as [flags](https://github.com/inoryy/tensorflow2-deep-reinforcement-learning/blob/master/a2c.py#L12-L17), e.g. `python a2c.py --batch_size=256`.
