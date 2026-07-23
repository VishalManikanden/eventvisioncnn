import numpy as np
from eventvision.models import build_cnn, compile_cnn


def test_build_cnn_output_shape():
    model = build_cnn(input_shape=(16, 16, 2), num_classes=5)
    model = compile_cnn(model)

    fake_batch = np.zeros((4, 16, 16, 2), dtype=np.float32)
    predictions = model(fake_batch)

    assert predictions.shape == (4, 5)
