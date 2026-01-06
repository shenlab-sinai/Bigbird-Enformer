"""
Helper functions adapted from enformer-pytorch/data.py from lucidrains
Adapted from enformer/enformer-training.ipynb from google deepmind
"""
import os
import json
import functools
import tensorflow as tf
import torch
import numpy as np
import torch
import torch.nn.functional as F

# Helper Function from enformer-pytorch
seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord('a')] = 0
seq_indices_embed[ord('c')] = 1
seq_indices_embed[ord('g')] = 2
seq_indices_embed[ord('t')] = 3
seq_indices_embed[ord('n')] = 4
seq_indices_embed[ord('A')] = 0
seq_indices_embed[ord('C')] = 1
seq_indices_embed[ord('G')] = 2
seq_indices_embed[ord('T')] = 3
seq_indices_embed[ord('N')] = 4
seq_indices_embed[ord('.')] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord('a')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('c')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('g')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('t')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('n')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('A')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('C')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('G')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('T')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('N')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('.')] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()

def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(map(lambda t: np.frombuffer(t.encode(), dtype = np.uint8).copy(), seq_strs))
    seq_chrs = list(map(torch.from_numpy, np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]

def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]

def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]

def seq_indices_to_one_hot(t, padding = -1):
    is_padding = t == padding
    t = t.clamp(min = 0)
    one_hot = F.one_hot(t, num_classes = 5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out

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