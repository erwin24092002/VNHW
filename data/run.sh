# python convert_uit2lmdb.py --root_dir data_raw/UIT_HWDB_line --outputPath data_lmdb/train --split train_data
# python convert_uit2lmdb.py --root_dir data_raw/UIT_HWDB_line --outputPath data_lmdb/test --split test_data
# python create_character_dict.py --root_dir data_raw/UIT_HWDB_line --output_path uit_hwdb_line.txt
python analyze.py --root_dir data_raw/UIT_HWDB_line
