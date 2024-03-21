

def test_remove_small_blobs():

    import numpy as np
    from ..src.evaluator.on_blobs import remove_small_blobs

    # Create a test binary mask with predefined blobs
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[10:50, 10:50] = 1  # Large blob
    test_mask[70:75, 70:75] = 1  # Small blob

    # Define the minimum size for blob removal
    min_size = 30

    # Process the test mask
    processed_mask = remove_small_blobs(np.array([test_mask]), min_size)[0]

    # Check that the large blob is still present
    assert processed_mask[30, 30] == 1, "Large blob should not be removed"

    # Check that the small blob has been removed
    assert processed_mask[72, 72] == 0, "Small blob should be removed"

    print("Test passed.")