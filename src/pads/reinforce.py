"""Optional TensorFlow 1-compatible LSTM REINFORCE implementation.

Training uses complete episodes so the recurrent state is recomputed from the
same observations used during rollout. See docs/implementation-notes.md for the
intentional corrections relative to the archived research prototype.
"""

from collections import deque
from pathlib import Path

import numpy as np

from .constants import NUM_ACTIONS, STATE_DIM
from .policy import discounted_returns, select_action


class PolicyGradientREINFORCE:
    def __init__(self, *, hidden_dim=32, learning_rate=0.0001, discount=0.9,
                 regularization=0.001, max_gradient=5.0, seed=0):
        try:
            import tensorflow as tensorflow
        except ImportError as exc:
            raise RuntimeError(
                "LSTM training requires TensorFlow 2.15 on Python 3.10 or 3.11. "
                "Install with: pip install -e '.[train]'. "
                "The simulate command runs without TensorFlow."
            ) from exc
        if not tensorflow.__version__.startswith("2.15."):
            raise RuntimeError("This compatibility implementation requires TensorFlow 2.15.x.")
        if hidden_dim < 1 or learning_rate <= 0 or max_gradient <= 0:
            raise ValueError("hidden_dim, learning_rate, and max_gradient must be positive")
        if not 0 <= discount <= 1 or regularization < 0:
            raise ValueError("discount must be in [0, 1] and regularization must be nonnegative")

        tf = tensorflow.compat.v1
        tf.disable_eager_execution()
        self.tf = tf
        self.hidden_dim = hidden_dim
        self.discount = discount
        self.rng = np.random.default_rng(seed)
        self.return_history = deque(maxlen=1_000_000)
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(seed)
            self.states = tf.placeholder(tf.float32, [1, None, STATE_DIM], "states")
            self.masks = tf.placeholder(tf.bool, [1, None, NUM_ACTIONS], "masks")
            self.initial_c = tf.placeholder(tf.float32, [1, hidden_dim], "initial_c")
            self.initial_h = tf.placeholder(tf.float32, [1, hidden_dim], "initial_h")
            self.actions = tf.placeholder(tf.int32, [1, None], "actions")
            self.returns = tf.placeholder(tf.float32, [1, None], "returns")
            with tf.variable_scope("policy_network"):
                cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, state_is_tuple=True)
                initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.initial_c, self.initial_h)
                output, self.next_recurrent = tf.nn.dynamic_rnn(
                    cell, self.states, initial_state=initial_state, dtype=tf.float32
                )
                weights = tf.get_variable("output_weights", [hidden_dim, NUM_ACTIONS],
                                          initializer=tf.glorot_uniform_initializer(seed=seed))
                bias = tf.get_variable("output_bias", [NUM_ACTIONS], initializer=tf.zeros_initializer())
                self.logits = tf.tensordot(output, weights, axes=1) + bias
            masked_logits = tf.where(self.masks, self.logits, tf.fill(tf.shape(self.logits), -1e9))
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.actions, logits=masked_logits
            )
            self.policy_loss = tf.reduce_mean(cross_entropy * self.returns)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")
            penalty = tf.add_n([tf.reduce_sum(tf.square(variable)) for variable in variables])
            self.loss = self.policy_loss + regularization * penalty
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9)
            gradients = [(gradient, variable) for gradient, variable
                         in optimizer.compute_gradients(self.loss) if gradient is not None]
            clipped, self.gradient_norm = tf.clip_by_global_norm(
                [gradient for gradient, _ in gradients], max_gradient
            )
            self.global_step = tf.train.get_or_create_global_step()
            self.train_op = optimizer.apply_gradients(
                list(zip(clipped, [variable for _, variable in gradients])),
                global_step=self.global_step,
            )
            self.saver = tf.train.Saver(max_to_keep=3)
            initialize = tf.global_variables_initializer()
        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        self.session = tf.Session(graph=self.graph, config=config)
        self.session.run(initialize)

    def initial_state(self):
        return (np.zeros((1, self.hidden_dim), dtype=np.float32),
                np.zeros((1, self.hidden_dim), dtype=np.float32))

    def act(self, state, recurrent, mask, *, greedy=False):
        logits, next_recurrent = self.session.run(
            [self.logits, self.next_recurrent],
            {self.states: np.asarray(state)[None, None, :],
             self.initial_c: recurrent[0], self.initial_h: recurrent[1]},
        )
        action = select_action(logits[0, 0], mask, self.rng, greedy=greedy)
        return action, tuple(np.array(value, copy=True) for value in next_recurrent)

    def update(self, trajectory):
        if not trajectory:
            raise ValueError("Cannot update the policy with an empty episode.")
        for step in trajectory:
            if not 0 <= step.action < NUM_ACTIONS or not step.mask[step.action]:
                raise ValueError("The rollout contains an action prohibited by its mask.")
        returns = discounted_returns([step.reward for step in trajectory], self.discount)
        self.return_history.extend(returns.tolist())
        history = np.asarray(self.return_history)
        returns = returns - history.mean()
        standard_deviation = history.std()
        if standard_deviation > 1e-8:
            returns /= standard_deviation
        initial_c, initial_h = self.initial_state()
        _, loss = self.session.run(
            [self.train_op, self.loss],
            {self.states: np.stack([step.state for step in trajectory])[None],
             self.masks: np.stack([step.mask for step in trajectory])[None],
             self.actions: [[step.action for step in trajectory]],
             self.returns: returns[None], self.initial_c: initial_c, self.initial_h: initial_h},
        )
        return float(loss)

    def save(self, directory):
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        with self.graph.as_default():
            return self.saver.save(self.session, str(directory / "policy"), global_step=self.global_step)

    def restore(self, path):
        path = Path(path).expanduser().resolve()
        checkpoint = self.tf.train.latest_checkpoint(str(path)) if path.is_dir() else str(path)
        if not checkpoint or not Path(checkpoint + ".index").is_file():
            raise ValueError(f"No TensorFlow policy checkpoint found at {path}.")
        self.saver.restore(self.session, checkpoint)

    def close(self):
        self.session.close()
