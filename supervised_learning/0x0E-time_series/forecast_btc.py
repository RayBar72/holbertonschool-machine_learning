#!/usr/bin/env python3
"""
Modulus that forecast a time series of BTC
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
cleaning = __import__('preprocess_data').cleans
preprocess = __import__('preprocess_data').train_val_test


class WindowGenerator():
    """Window generatos
    """
    def __init__(self, input_width, label_width, shift, train_df=train_df,
                 val_df=val_df, test_df=test_df, label_columns=None):
        """Window generator initializer
        """
        # Raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        # Label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i,
                               name in enumerate(train_df.columns)}
        # Window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.
                                       total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.
                                       total_window_size)[self.labels_slice]

    def __repr__(self):
        """
        Representation
        """
        return '\n'.join([
            'Total window size: {}'.format(self.total_window_size),
            'Input indices: {}'.format(self.input_indices),
            'Label indices: {}'.format(self.label_indices),
            'Label column name(s): {}'.format(self.label_columns)])

    def split_window(self, features):
        """
        Splits windows
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]]
                               for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """
        Makes datasets
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        """
        Trains
        """
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """
        Returs validation dataset
        """
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """
        Returns testdataset
        """
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        """
        Plots results
        """
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel('{} [normed]'.format(plot_col))
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col,
                                                                 None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices,
                            predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Close')


def compile_and_fit(model, window, patience=2):
    """
    Compiles and executs fitting
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss='mean_squared_error',
                  optimizer='Adam',
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train,
                        epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


if __name__ == '__main__':
    filename = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
    df = cleaning(filename)
    train_df, val_df, test_df = preprocess(df)
    MAX_EPOCHS = 2000

    wide_window = WindowGenerator(24,
                                  24,
                                  1,
                                  label_columns=['Close'])

    lstm = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(1)
    ])

    history = compile_and_fit(lstm, wide_window)
    val_perf = {}
    perf = {}
    val_perf['LSTM'] = lstm.evaluate(wide_window.val)
    perf['LSTM'] = lstm.evaluate(wide_window.test, verbose=True)

    wide_window.plot(lstm)
