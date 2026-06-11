#pragma once
#include <string>
#include "utils.h"

Dataset load_csv(std::string dataset_name, std::string path, int input_size, int label_size);
void save_dataset(std::string output_path, Dataset dataset);
Dataset load_dataset(std::string file_path);
bool dataset_exists(std::string file_path);
Dataset get_dataset(std::string dataset_name, std::string csv_path, std::string save_path, int input_size, int label_size);