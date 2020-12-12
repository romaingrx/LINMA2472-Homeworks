from typing import Dict

import numpy as np
import tensorflow as tf

import utils

class ExecuteEveryNExamplesCallback(tf.keras.callbacks.Callback):
    """
    Executes a given function approximately every N examples, depending on if the period is an even multiple of the batch size or not.
    """
    # Taken from https://github.com/lebrice/blurred-GAN/blob/7bc2e2e8dece39073e353a3fbada56bfdacc7430/callbacks.py

    def __init__(self, n: int, starting_from: int = 0):
        """
        args:
            n: executes the `self.function(batch, logs)` method approximately every N examples
            starting_from: The first invocation should occur after this number of examples (defaults to 0)
        """
        super().__init__()
        self.period = n
        self.num_invocations = 0
        self.samples_seen = 0
        self.starting_from = starting_from

    def on_batch_end(self, batch, logs: Dict):
        batch_size = logs["size"]
        self.samples_seen += batch_size
        i = (self.samples_seen - self.starting_from) // self.period
        # print("\n", i, self.samples_seen, self.starting_from, self.period)
        if self.samples_seen < self.starting_from:
            return
        if i >= self.num_invocations:
            self.num_invocations += 1
            # print(f"\nsamples_seen: {self._samples_seen}, batch: {batch}, i: {self.i}\n")
            # TODO: Check the function signature.
            self.function(batch, logs)

    def function(self, batch, logs):
        raise NotImplementedError("Implement the 'function' inside your class!")

class SaveSampleGridCallback(ExecuteEveryNExamplesCallback):
    def __init__(self, log_dir: str, every_n_examples=1000):
        self.log_dir = log_dir
        super().__init__(n=every_n_examples)

    def function(self, batch, logs):
        self.make_grid()

    def make_grid(self, *args):
        from train2 import train_data
        for t in train_data.take(1):
            samples = t[:64]
            break

        samples = utils.normalize_images(samples)
        figure = utils.samples_grid(samples)  # TODO: write figure to a file?
        figure.savefig(self.log_dir + f"/samples_grid_{self.samples_seen:06}.png")
        image = utils.plot_to_image(figure)
        with self.model.summary_writer.as_default():
            tf.summary.image("samples_grid_real", image, step=self.num_invocations)

class GenerateSampleGridCallback(ExecuteEveryNExamplesCallback):
    def __init__(self, log_dir: str, show_blurred_samples=False, every_n_examples=1000, also_save_files=True):
        self.log_dir = log_dir
        self.show_blurred_samples = show_blurred_samples
        super().__init__(n=every_n_examples)

        self.also_save_files = also_save_files

        # we need a constant random vector which will not change over the course of training. 
        self.latents: np.ndarray = None

    def function(self, batch, logs):
        self.make_grid()

    def on_train_begin(self, logs: Dict):
        self.latents = tf.random.uniform([64, self.model.generator.input_shape[-1]])

    def make_grid(self, *args):
        samples = self.model.generate_samples(self.latents, training=False)
        if self.show_blurred_samples:
            samples = self.model.blur(samples)

        samples = utils.normalize_images(samples)
        figure = utils.samples_grid(samples)  # TODO: write figure to a file?
        figure.savefig(self.log_dir + f"/samples_grid_{self.samples_seen:06}.png")
        image = utils.plot_to_image(figure)
        with self.model.summary_writer.as_default():
            tf.summary.image("samples_grid", image, step=self.num_invocations)

class SaveModelCallback(ExecuteEveryNExamplesCallback):
    def __init__(self, checkpoint_manager: tf.train.CheckpointManager, n: int = 10_000):
        super().__init__(n=n)
        self.manager = checkpoint_manager

    def function(self, batch, logs):
        self.manager.save(self.samples_seen)

class LogMetricsCallback(ExecuteEveryNExamplesCallback):
    def __init__(self, every_n_examples: int = 100):
        super().__init__(n=every_n_examples)

    def on_train_begin(self, logs):
        self.samples_seen = self.model.n_img.numpy()

    def function(self, batch: int, logs: Dict):
        self.write_metric_summaries(logs, prefix="batch_")

    def on_epoch_end(self, epoch: int, logs: Dict):
        self.write_metric_summaries(logs, prefix="epoch_")

    def write_metric_summaries(self, logs: Dict, prefix="", flush=False):
        with self.model.summary_writer.as_default():
            for name, value in logs.items():
                if name not in ("batch", "size"):
                    tf.summary.scalar(f"{prefix}{name}", value, step=self.num_invocations)
            if flush:
                self.model.summary_writer.flush()
