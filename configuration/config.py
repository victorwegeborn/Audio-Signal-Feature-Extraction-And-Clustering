# List all classes that are to be included in the clustering.
# Be sure to use the exact capitalization from the subfolders
# in the sample directory
classes = [
    'cymbal',
    'Kick',
    'Snare',
    'Tom',
    'hh',
]

# List features to include in the training data
# that have been processed and stored on disc
# features available: mfcc, decay, brightness, rolloff
features = [
    'mfcc',
    'decay',
    'brightness',
    'rolloff'
]
