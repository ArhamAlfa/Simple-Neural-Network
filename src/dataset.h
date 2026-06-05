#pragma once
#include <string>
#include "utils.h"

Dataset load_csv(std::string dataset_name, std::string path, int input_size, int label_size);
static void save_vector(std::ofstream &file, Eigen::VectorXd &vector);
static Eigen::VectorXd load_vector(std::ifstream &file);
static void write_string(std::ofstream &file, std::string string);
static std::string read_string(std::ifstream &file);
static void save_metadata(std::ofstream &file, Dataset &dataset);
static void load_metadata(std::ifstream &file, Dataset &dataset, int metadata_size);
void save_dataset(std::string output_path, Dataset dataset);
Dataset load_dataset(std::string file_path);
bool dataset_exists(std::string file_path);
Dataset get_dataset(std::string dataset_name, std::string csv_path, std::string save_path, int input_size, int label_size);
void print_dataset(Dataset &dataset);