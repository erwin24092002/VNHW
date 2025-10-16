python convert_uit2lmdb.py --root_dir data_raw/UIT_HWDB_word --outputPath data_lmdb/train --split train_data
python convert_uit2lmdb.py --root_dir data_raw/UIT_HWDB_word --outputPath data_lmdb/test --split test_data
python create_character_dict.py --root_dir data_raw/UIT_HWDB_word --output_path uit_hwdb_word.txt
python analyze.py --root_dir data_raw/UIT_HWDB_word