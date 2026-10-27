"""Which edge columns message passing and the readout see. The seven files are conf/features/cfg1.yaml … cfg7.yaml."""


def feature_pair(features):
    """{'mp', 'readout'} from the selected features config."""
    return {'mp': str(features['mp']), 'readout': str(features['readout'])}
