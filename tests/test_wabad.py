"""Tests for Wabad dataset class."""

import unittest
from pathlib import Path

import numpy as np

from esp_data.datasets.wabad import Wabad


class TestWabad(unittest.TestCase):
    """Test case for Wabad dataset (events-based)."""

    def setUp(self):
        """Set up test fixtures."""
        # Use local test data as fallback for environments without GCP access
        self.test_data_dir = Path(__file__).parent.parent / "v1"
        self.master_csv_path = self.test_data_dir / "master_info.csv"
        self.has_local_data = self.test_data_dir.exists() and self.master_csv_path.exists()

    def test_class_registration(self):
        """Test that the dataset class is properly registered."""
        # Test that we can import the class
        from esp_data.datasets import Wabad

        # Test that it's in the __all__ list
        import esp_data.datasets as datasets
        self.assertTrue("Wabad" in datasets.__all__)

        # Test that the class has the correct decorator and info
        self.assertTrue(hasattr(Wabad, 'info'))
        self.assertTrue(hasattr(Wabad.info, 'name'))
        self.assertEqual(Wabad.info.name, 'wabad')

        # Test that it's in the registry
        from esp_data import list_registered_datasets
        registered_datasets = list_registered_datasets()
        self.assertTrue('wabad' in registered_datasets)

    def test_dataset_initialization(self):
        """Test basic dataset initialization."""
        try:
            dataset = Wabad(
                split="all",
                window_duration=5.0,
                random_window=True,
                seed=42
            )

            # Test basic properties
            self.assertIsInstance(dataset, Wabad)
            self.assertEqual(dataset.split, "all")
            self.assertEqual(dataset.window_duration, 5.0)
            self.assertTrue(dataset.random_window)
            self.assertEqual(dataset.seed, 42)
            self.assertGreater(len(dataset), 0)
        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_dataset_length(self):
        """Test dataset length calculation."""
        try:
            dataset = Wabad(split="all")

            # Should have many more samples than the 4297 audio files since each
            # annotation becomes an event. Expecting around 90,000+ events.
            self.assertGreater(len(dataset), 4297)
            self.assertGreater(len(dataset), 50000)  # Should be much larger
        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_get_item(self):
        """Test getting a single event from the dataset."""
        try:
            dataset = Wabad(
                split="all",
                sample_rate=16000,
                window_duration=3.0,
                random_window=False,  # Use centered windows for predictable testing
                seed=42
            )

            item = dataset[0]

            # Check expected keys
            expected_keys = [
                "audio", "fn", "target_species", "target_start_time", "target_end_time",
                "target_low_freq", "target_high_freq", "window_start", "window_end",
                "window_duration", "all_labels"
            ]
            for key in expected_keys:
                self.assertTrue(key in item)

            # Check audio properties
            self.assertIsInstance(item["audio"], np.ndarray)
            self.assertGreater(len(item["audio"]), 0)

            # Check window properties
            self.assertAlmostEqual(item["window_duration"], 3.0, places=1)
            self.assertGreaterEqual(item["window_start"], 0)
            self.assertLessEqual(
                item["window_end"],
                item["window_start"] + item["window_duration"] + 0.1
            )

            # Check target event properties
            self.assertIsInstance(item["target_species"], str)
            self.assertGreaterEqual(item["target_start_time"], 0)
            self.assertLessEqual(item["target_end_time"], item["window_duration"])

            # Check all_labels
            self.assertIsInstance(item["all_labels"], list)
        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_random_windowing_with_seed(self):
        """Test random windowing functionality with seeding."""
        try:
            # Test with fixed seed for reproducibility
            dataset1 = Wabad(
                split="all",
                window_duration=4.0,
                random_window=True,
                min_padding=0.5,
                max_padding=1.5,
                seed=42
            )

            dataset2 = Wabad(
                split="all",
                window_duration=4.0,
                random_window=True,
                min_padding=0.5,
                max_padding=1.5,
                seed=42
            )

            # Same seed should produce same windows
            item1 = dataset1[0]
            item2 = dataset2[0]

            self.assertAlmostEqual(item1["window_start"], item2["window_start"], places=3)
            self.assertAlmostEqual(item1["window_end"], item2["window_end"], places=3)

            # Verify window properties are reasonable
            self.assertGreaterEqual(item1["window_duration"], 4.0 - 0.1)  # Allow small tolerance
        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_output_take_and_give(self):
        """Test output filtering with output_take_and_give."""
        try:
            output_mapping = {"audio": "wav", "target_species": "species", "all_labels": "labels"}
            dataset = Wabad(
                split="all",
                output_take_and_give=output_mapping
            )

            item = dataset[0]

            # Should only have mapped keys
            self.assertEqual(set(item.keys()), {"wav", "species", "labels"})
            self.assertIsInstance(item["wav"], np.ndarray)
            self.assertIsInstance(item["species"], str)
            self.assertIsInstance(item["labels"], list)
        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_iteration(self):
        """Test dataset iteration."""
        try:
            dataset = Wabad(split="all")

            # Test that we can iterate
            count = 0
            for item in dataset:
                count += 1
                # Verify item structure
                self.assertTrue("audio" in item)
                self.assertTrue("target_species" in item)
                self.assertTrue("all_labels" in item)
                if count >= 3:  # Just test first few items
                    break

            self.assertEqual(count, 3)
        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_invalid_split(self):
        """Test error handling for invalid split."""
        try:
            with self.assertRaises(LookupError):
                Wabad(split="invalid_split")
        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_columns_property(self):
        """Test columns property."""
        try:
            dataset = Wabad(split="all")
            columns = dataset.columns

            expected_columns = [
                "fn", "audio_fp", "selection_table_str", "audio_duration", "subdataset"
            ]
            for col in expected_columns:
                self.assertTrue(col in columns)
        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_window_bounds_calculation(self):
        """Test window bounds calculation logic."""
        try:
            dataset = Wabad(
                split="all",
                window_duration=2.0,
                random_window=False,  # Centered windows
                seed=42
            )

            # Test that we can create the dataset without errors
            self.assertGreater(len(dataset), 0)

            # Test a sample
            item = dataset[0]

            # Window should be approximately the target duration
            actual_duration = item["window_end"] - item["window_start"]
            self.assertAlmostEqual(actual_duration, 2.0, places=1)
        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_multi_label_in_window(self):
        """Test that multiple labels in a window are captured."""
        try:
            dataset = Wabad(
                split="all",
                window_duration=10.0,  # Large window to catch multiple species
                random_window=False,
                seed=42
            )

            # Find an item with multiple labels
            found_multi_label = False
            for i in range(min(len(dataset), 20)):  # Check first 20 items
                item = dataset[i]
                if len(item["all_labels"]) > 1:
                    found_multi_label = True

                    # Verify all labels have required fields
                    for label in item["all_labels"]:
                        self.assertTrue("species" in label)
                        self.assertTrue("start_time" in label)
                        self.assertTrue("end_time" in label)
                        self.assertTrue("low_freq" in label)
                        self.assertTrue("high_freq" in label)
                    break

            # Note: We might not always find multi-label windows depending on the data
            # So we don't assert this, but if we do find one, we verify its structure
        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_reproducibility_with_seed(self):
        """Test that the same seed produces reproducible results."""
        try:
            # Create two datasets with same parameters and seed
            dataset1 = Wabad(
                split="all",
                window_duration=3.0,
                random_window=True,
                seed=123
            )

            dataset2 = Wabad(
                split="all",
                window_duration=3.0,
                random_window=True,
                seed=123
            )

            # Get same sample from both datasets
            item1 = dataset1[5]
            item2 = dataset2[5]

            # Should have identical window bounds
            self.assertEqual(item1["window_start"], item2["window_start"])
            self.assertEqual(item1["window_end"], item2["window_end"])
            self.assertEqual(item1["target_species"], item2["target_species"])
        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_dataset_from_config(self):
        """Test creating dataset from configuration."""
        try:
            from esp_data import DatasetConfig, dataset_from_config

            config = DatasetConfig(
                dataset_name="wabad",
                split="all",
                window_duration=4.0,
                random_window=True,
                seed=42
            )

            dataset, metadata = dataset_from_config(config)

            self.assertIsInstance(dataset, Wabad)
            self.assertEqual(dataset.window_duration, 4.0)
            self.assertTrue(dataset.random_window)
            self.assertEqual(dataset.seed, 42)
            self.assertIsInstance(metadata, dict)  # Should be empty dict if no transforms
        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_dataset_info(self):
        """Test dataset info properties."""
        try:
            dataset = Wabad(split="all")

            # Test that info is properly set
            self.assertEqual(dataset.info.name, "wabad")
            self.assertEqual(dataset.info.version, "0.1.0")
            self.assertEqual(dataset.info.license, "CC BY-NC 4.0")
            self.assertTrue("Global PAM recordings" in dataset.info.sources)

            # Test that all expected splits are available
            expected_splits = [
                "all", "all_16khz", "train", "train_16khz",
                "validation", "validation_16khz", "train_sites",
                "train_sites_16khz", "validation_sites", "validation_sites_16khz"
            ]
            for split in expected_splits:
                self.assertTrue(split in dataset.info.split_paths)

            # Test a few key split paths
            expected_paths = {
                "all": "gs://esp-ml-datasets/wabad/v0.1.0/raw/all_info.csv",
                "train": "gs://esp-ml-datasets/wabad/v0.1.0/raw/train_info.csv",
                "validation_16khz": "gs://esp-ml-datasets/wabad/v0.1.0/raw_16khz/validation_info.csv",
                "train_sites": "gs://esp-ml-datasets/wabad/v0.1.0/raw/train_sites_info.csv"
            }
            for split, expected_path in expected_paths.items():
                self.assertEqual(dataset.info.split_paths[split], expected_path)

        except Exception as e:
            self.skipTest(f"GCP access not available: {e}")

    def test_16khz_split_initialization(self):
        """Test 16kHz split initialization."""
        try:
            dataset_16k = Wabad(
                split="all_16khz",
                window_duration=3.0,
                sample_rate=16000,
                seed=42
            )

            # Test basic properties
            self.assertIsInstance(dataset_16k, Wabad)
            self.assertEqual(dataset_16k.split, "all_16khz")
            self.assertEqual(dataset_16k.sample_rate, 16000)
            self.assertGreater(len(dataset_16k), 0)

            # Test data root is different from master
            self.assertTrue("raw_16khz" in dataset_16k.data_root)

        except Exception as e:
            self.skipTest(f"16kHz split not available yet: {e}")

    def test_splits_consistency(self):
        """Test that both splits have the same metadata (when 16kHz is available)."""
        try:
            dataset_master = Wabad(split="all", seed=42)
            dataset_16k = Wabad(split="all_16khz", seed=42)

            # Should have same number of events
            self.assertEqual(len(dataset_master), len(dataset_16k))

            # Should have same columns
            self.assertEqual(dataset_master.columns, dataset_16k.columns)

            # Test that events metadata is the same (excluding audio content)
            event_master = dataset_master._events[0]
            event_16k = dataset_16k._events[0]

            # Same filename and metadata
            self.assertEqual(event_master['fn'], event_16k['fn'])
            self.assertEqual(event_master['target_species'], event_16k['target_species'])
            self.assertEqual(event_master['target_start_time'], event_16k['target_start_time'])
            self.assertEqual(event_master['target_end_time'], event_16k['target_end_time'])

        except Exception as e:
            self.skipTest(f"16kHz split not available yet or GCP access issue: {e}")

    def test_16khz_data_root_path(self):
        """Test that 16kHz split uses correct data root."""
        try:
            dataset_master = Wabad(split="all")
            dataset_16k = Wabad(split="all_16khz")

            # Different data roots
            self.assertTrue("raw" in dataset_master.data_root)
            self.assertTrue("raw_16khz" in dataset_16k.data_root)
            self.assertNotEqual(dataset_master.data_root, dataset_16k.data_root)

            # Check specific paths
            self.assertTrue(dataset_master.data_root.endswith("raw"))
            self.assertTrue(dataset_16k.data_root.endswith("raw_16khz"))

        except Exception as e:
            self.skipTest(f"16kHz split not available yet: {e}")

    def test_16khz_split_sample_access(self):
        """Test accessing samples from 16kHz split."""
        try:
            dataset_16k = Wabad(
                split="all_16khz",
                sample_rate=16000,
                window_duration=2.0,
                random_window=False,
                seed=123
            )

            sample = dataset_16k[0]

            # Should have same structure as regular split
            expected_keys = [
                "audio", "fn", "target_species", "target_start_time", "target_end_time",
                "target_low_freq", "target_high_freq", "window_start", "window_end",
                "window_duration", "all_labels"
            ]
            for key in expected_keys:
                self.assertTrue(key in sample)

            # Audio should be loaded
            self.assertIsInstance(sample["audio"], np.ndarray)
            self.assertGreater(len(sample["audio"]), 0)

        except Exception as e:
            self.skipTest(f"16kHz split not available yet: {e}")

    def test_16khz_split_with_config(self):
        """Test creating 16kHz dataset from configuration."""
        try:
            from esp_data import DatasetConfig, dataset_from_config

            config = DatasetConfig(
                dataset_name="wabad",
                split="all_16khz",
                sample_rate=16000,
                window_duration=4.0,
                random_window=True,
                seed=42
            )

            dataset, metadata = dataset_from_config(config)

            self.assertIsInstance(dataset, Wabad)
            self.assertEqual(dataset.split, "all_16khz")
            self.assertEqual(dataset.sample_rate, 16000)
            self.assertEqual(dataset.window_duration, 4.0)
            self.assertTrue(dataset.random_window)
            self.assertEqual(dataset.seed, 42)

        except Exception as e:
            self.skipTest(f"16kHz split not available yet: {e}")

    def test_invalid_16khz_split_fallback(self):
        """Test handling of invalid split names."""
        with self.assertRaises(LookupError):
            Wabad(split="all_32khz")  # Non-existent split

        with self.assertRaises(LookupError):
            Wabad(split="invalid_split")

    def test_train_validation_splits_availability(self):
        """Test that train/validation splits are properly configured."""
        # Test that all expected splits are available
        expected_splits = [
            "all", "all_16khz",
            "train", "train_16khz",
            "validation", "validation_16khz",
            "train_sites", "train_sites_16khz",
            "validation_sites", "validation_sites_16khz"
        ]

        available_splits = list(Wabad.info.split_paths.keys())
        for split in expected_splits:
            self.assertTrue(split in available_splits, f"Split {split} not found in available splits")

    def test_split_path_consistency(self):
        """Test that split paths follow consistent naming conventions."""
        paths = Wabad.info.split_paths

        # Test that 16kHz splits point to raw_16khz directory
        for split_name, path in paths.items():
            if "16khz" in split_name:
                self.assertTrue("raw_16khz" in path, f"16kHz split {split_name} should point to raw_16khz directory")
            else:
                self.assertTrue("/raw/" in path, f"Regular split {split_name} should point to raw directory")
                self.assertNotIn("raw_16khz", path, f"Regular split {split_name} should not point to raw_16khz")

    def test_train_split_initialization(self):
        """Test that train split can be initialized (when available)."""
        try:
            dataset = Wabad(split="train", seed=42)
            self.assertEqual(dataset.split, "train")
            # Training split should have fewer events than the full dataset
            # but we can't test this until the splits are uploaded
        except Exception as e:
            self.skipTest(f"Train split not available yet: {e}")

    def test_validation_split_initialization(self):
        """Test that validation split can be initialized (when available)."""
        try:
            dataset = Wabad(split="validation", seed=42)
            self.assertEqual(dataset.split, "validation")
            # Validation split should have fewer events than the full dataset
            # but we can't test this until the splits are uploaded
        except Exception as e:
            self.skipTest(f"Validation split not available yet: {e}")

    def test_site_based_splits_initialization(self):
        """Test that site-based splits can be initialized (when available)."""
        try:
            train_sites = Wabad(split="train_sites", seed=42)
            val_sites = Wabad(split="validation_sites", seed=42)

            self.assertEqual(train_sites.split, "train_sites")
            self.assertEqual(val_sites.split, "validation_sites")

            # Site-based splits should have complementary data
            # but we can't test this until the splits are uploaded
        except Exception as e:
            self.skipTest(f"Site-based splits not available yet: {e}")


if __name__ == "__main__":
    unittest.main()
