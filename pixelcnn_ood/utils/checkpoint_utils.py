"""Utilities for Tensorboard logging, checkpointing and Log Likelihood computation."""


import os
import tempfile
import time
from absl import logging
import tensorflow as tf

gfile = tf.io.gfile
# from google3.third_party.tensorflow.python.ops import summary_ops_v2  # pylint: disable=g-direct-tensorflow-import


class CNSModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
  """Wrapper for callback class to save checkpoints to CNS."""

  def _save_model(self, epoch, logs):
    """Saves the model.

    Arguments:
        epoch: the epoch this iteration is in.
        logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
    """
    logs = logs or {}

    if isinstance(self.save_freq,
                  int) or self.epochs_since_last_save >= self.period:
      self.epochs_since_last_save = 0
      filepath = self._get_file_path(epoch, None, logs)

      if self.save_best_only:
        current = logs.get(self.monitor)
        if current is None:
          logging.warning(
              'Can save best model only with %s available, skipping.',
              self.monitor)
        else:
          if self.monitor_op(current, self.best):
            if self.verbose > 0:
              logging.info(
                  '\nEpoch %d: %s improved from %f to %f, saving model to %s',
                  epoch+1, self.monitor, self.best, current, filepath)

            self.best = current
            if self.save_weights_only:
              save_model_to_cns(self.model, filepath, include_optimizer=False)
            else:
              save_model_to_cns(self.model, filepath)
          else:
            if self.verbose > 0:
              logging.info(
                  '\nEpoch %d: %s did not improve from %f',
                  epoch+1, self.monitor, self.best)
      else:
        if self.verbose > 0:
          logging.info('\nEpoch %d: saving model to %s', epoch+1, filepath)
        if self.save_weights_only:
          save_model_to_cns(self.model, filepath, include_optimizer=False)
        else:
          save_model_to_cns(self.model, filepath)

      self._maybe_remove_file()


def save_model(model, filepath, overwrite=False):
  """Saves a model, buffering in a local file.

  The model is saved on the local or remote Borg scratch disk. The file
  is then copied to the path specified by the user.

  Args:
    model: Keras model instance to be saved.
    filepath: The path to which the model is copied. The directory must already
      exist.
    overwrite: passed to gfile.Copy.

  Raises:
      ImportError: if h5py is not available.
  """

  local_filename = filepath.split('/')[-1]
  parent = tempfile.gettempdir()
  tmpfileprefix = str(int(time.time())) + '_' + local_filename
  tmp_filename = os.path.join(parent, tmpfileprefix)
  print(tmp_filename)
  model.save_weights(tmp_filename)
  if os.path.exists(tmp_filename):
    gfile.copy(tmp_filename, filepath, overwrite=overwrite)
    gfile.remove(tmp_filename)
  else:
    for file in os.listdir(parent):
      if file.startswith(tmpfileprefix):
        gfile.copy(
            os.path.join(parent, file),
            os.path.join(os.path.abspath(os.path.join(filepath, os.pardir)),
                         local_filename+file[len(tmpfileprefix):]),
            overwrite=overwrite)
        gfile.remove(os.path.join(parent, file))


def save_model_to_cns(model, filepath, overwrite=True, include_optimizer=True):
  # Check if the user's filepath is a valid
  # if gfile.pywrapfile.FileName(filepath).is_local():
  #   raise ValueError('You must provide a valid GoogleFile path to save model '
  #                    'checkpoints.')
  include_optimizer = include_optimizer and model.optimizer
  save_model(model, filepath, overwrite=overwrite)