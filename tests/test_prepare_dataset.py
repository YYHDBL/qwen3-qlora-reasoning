from src.dataset_classifier import Classification, classify_record
from src.dataset_splitters import allocate_split_quotas


def test_prepare_dataset_support_modules_are_importable():
    classification = classify_record(
        (
            "In Alice's Wonderland, secret encryption rules are used on "
            "text. Here are some examples:\n"
            "\n"
            "abc -> bcd\n"
            "cat -> dbu\n"
            "zoo -> app\n"
            "Now, decrypt the following text: ifmmp"
        ),
        "hello",
    )

    quotas = allocate_split_quotas({"cipher": 10}, seed=42)

    assert isinstance(classification, Classification)
    assert classification.task_type == "cipher"
    assert quotas == {
        "train": {"cipher": 8},
        "validation": {"cipher": 1},
        "test": {"cipher": 1},
    }
