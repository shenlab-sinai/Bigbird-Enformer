"""
Adapted from enformer/enformer-training.ipynb from google deepmind
"""
import os
import json
import functools
import tensorflow as tf
import torch
import numpy as np

# Official DeepMind Helper Functions
def organism_path(organism):
    return os.path.join('gs://basenji_barnyard/data', organism)

def get_metadata(organism):
    path = os.path.join(organism_path(organism), 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)

def tfrecord_files(organism, subset):
    return sorted(tf.io.gfile.glob(os.path.join(
        organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
    )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

def deserialize(serialized_example, metadata):
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target, (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return {'sequence': sequence, 'target': target}

def get_dataset(organism, subset, num_threads=8):
    metadata = get_metadata(organism)
    dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset),
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_threads)
    dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                          num_parallel_calls=num_threads)
    return dataset

# PyTorch Bridge
class TFRecordDataLoader:
    """
    Wraps the TensorFlow dataset to yield PyTorch Tensors
    """
    def __init__(self, organism='human', subset='train', batch_size=1):
        self.organism = organism
        # Create the TF dataset
        self.tf_dataset = get_dataset(organism, subset)
        
        # Batch and Prefetch (Official Config)
        self.tf_dataset = self.tf_dataset.batch(batch_size).prefetch(2)
        
        # Create an iterator
        self.iterator = iter(self.tf_dataset)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
            seq = torch.from_numpy(batch['sequence'].numpy())
            target = torch.from_numpy(batch['target'].numpy())
            
            return seq, target
        except StopIteration:
            raise StopIteration