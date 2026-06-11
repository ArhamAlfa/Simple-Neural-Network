#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "dataset.h"

static void print_dataset(Dataset &dataset)
{
    std::cout << "\n\n-- PRINTING DATASET ---------" << std::endl;
    std::cout << "-- Dataset: " << std::get<std::string>(dataset.metadata["dataset_name"]) << std::endl;
    std::cout << "-- Path: " << std::get<std::string>(dataset.metadata["path"]) << std::endl;
    std::cout << std::get<std::string>(dataset.metadata["columns"]) << std::endl;

    for (int i = 0; i < dataset.inputs.size(); i++)
    {
        std::cout << dataset.inputs[i].transpose() << " " << dataset.labels[i].transpose() << std::endl;
    }
}

Dataset load_csv(std::string dataset_name, std::string path, int input_size, int label_size)
{
    std::cout << "Running load_csv()" << std::endl;

    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
        return {};
    }

    std::cout << "file opened." << std::endl;

    std::string line;
    Dataset dataset;
    dataset.metadata["dataset_name"] = dataset_name;
    dataset.metadata["path"] = path;

    // Save columns as metadata
    std::getline(file, line);
    dataset.metadata["columns"] = line;

    std::cout << "columns saved." << std::endl;

    // Read the file line by line
    while (std::getline(file, line))
    {
        // std::vector<std::string> row;
        std::stringstream ss(line);
        std::string value;

        Eigen::VectorXd input(input_size, 1);

        // Get all values till input size
        for (int i = 0; i < input_size; i++)
        {
            std::getline(ss, value, ',');
            input(i) = std::stod(value);
        }

        Eigen::VectorXd label(label_size, 1);

        // The rest of the stream are labels
        for (int i = 0; i < label_size; i++)
        {
            std::getline(ss, value, ',');
            label(i) = std::stod(value);
        }

        dataset.inputs.push_back(input);
        dataset.labels.push_back(label);
    }

    file.close();

    print_dataset(dataset);

    return dataset;
}

static void save_vector(std::ofstream &file, Eigen::VectorXd &vector)
{
    // std::cout << "Running save_vector()" << std::endl;
    int size = vector.size();

    // Write the size of the vector itself
    file.write((char *)&size, sizeof(int));

    // Write the vector down
    file.write((char *)vector.data(), size * sizeof(double));
}

static Eigen::VectorXd load_vector(std::ifstream &file)
{
    // std::cout << "Running load_vector()" << std::endl;
    int size;

    // Read and store size of vector into size
    file.read((char *)&size, sizeof(int));

    // Create vector of size size
    Eigen::VectorXd vector(size);

    // Read the bytes of array data and load it into vector
    file.read((char *)vector.data(), size * sizeof(double));

    return vector;
}

static void write_string(std::ofstream &file, std::string string)
{
    // std::cout << "Running write_string()" << std::endl;
    int size = string.size();

    // Write the size of the string
    file.write((char *)&size, sizeof(int));

    // Write the actual string contents
    // We don't need to cast string.data() with char* since a string is already an array of characters
    // So string.data() IS a pointer to an array of adjacent characters
    // Likewise string.size() already returns the amount of characters it uses so no need for sizeof()
    file.write(string.data(), string.size());
}

static std::string read_string(std::ifstream &file)
{
    // std::cout << "Running read_string()" << std::endl;
    int size;

    file.read((char *)&size, sizeof(int));

    // Create string of size with null characters
    std::string string(size, '\0');
    file.read(string.data(), size);

    return string;
}

static void save_metadata(std::ofstream &file, Dataset &dataset)
{
    // std::cout << "Running save_metadata()" << std::endl;
    // unpack the metadata map into [key, value] pairs similar to what you can do in Python
    for (auto &[key, value] : dataset.metadata)
    {
        std::cout << "-- Saving key: " << &key << std::endl;
        write_string(file, key);

        std::cout << "-- Saving value: " << &value << std::endl;

        // Get my value type by index according to how I defined variant<>
        int type_index = value.index();
        file.write((char *)&type_index, sizeof(int));

        int vi;
        double vd;

        switch (type_index)
        {
        // case int
        case 0:
            vi = std::get<int>(value);
            file.write((char *)&vi, sizeof(int));
            break;

        // case double
        case 1:
            vd = std::get<double>(value);
            file.write((char *)&vd, sizeof(double));
            break;

        // case string
        case 2:
            write_string(file, std::get<std::string>(value));
            break;

        default:
            break;
        }
    }
}

static void load_metadata(std::ifstream &file, Dataset &dataset, int metadata_size)
{
    // std::cout << "Running load_metadata()" << std::endl;
    for (int i = 0; i < metadata_size; i++)
    {
        std::string key = read_string(file);

        int type_index;
        file.read((char *)&type_index, sizeof(int));

        int vi;
        double vd;

        switch (type_index)
        {
        // case int
        case 0:
            file.read((char *)&vi, sizeof(int));
            dataset.metadata[key] = vi;
            break;

        // case double
        case 1:
            file.read((char *)&vd, sizeof(double));
            dataset.metadata[key] = vd;
            break;

        // case string
        case 2:
            dataset.metadata[key] = read_string(file);
            break;

        default:
            break;
        }
    }
}

void save_dataset(std::string output_path, Dataset dataset)
{
    std::cout << "Running save_dataset()" << std::endl;
    std::ofstream file(output_path, std::ios::binary);

    // Inidicate how many samples are in the dataset
    int size = dataset.inputs.size();
    file.write((char *)&size, sizeof(int));

    // Save each vector input and vector label as back-to-back pairings
    for (int i = 0; i < size; i++)
    {
        save_vector(file, dataset.inputs[i]);
        save_vector(file, dataset.labels[i]);
    }

    // Indicate how many metadata fields there are
    int metadata_size = dataset.metadata.size();
    file.write((char *)&metadata_size, sizeof(int));

    // Save metadata
    save_metadata(file, dataset);

    std::cout << "\n\n--- Dataset has been successfully created at: " << output_path << std::endl;
}

Dataset load_dataset(std::string file_path)
{
    std::cout << "Running load_dataset()" << std::endl;
    Dataset dataset;
    std::ifstream file(file_path, std::ios::binary);

    // Read how many samples in the dataset
    int size;
    file.read((char *)&size, sizeof(int));

    for (int i = 0; i < size; i++)
    {
        dataset.inputs.push_back(load_vector(file));
        dataset.labels.push_back(load_vector(file));
    }

    // Read size of dataset metadata
    int metadata_size;
    file.read((char *)&metadata_size, sizeof(int));

    // Load metadata
    load_metadata(file, dataset, metadata_size);

    std::cout << "\n\n--- Dataset has been successfully loaded." << std::endl;
    return dataset;
}

bool dataset_exists(std::string file_path)
{
    // std::cout << "Running dataset_exists()" << std::endl;
    std::ifstream file(file_path);
    return file.is_open();
}

Dataset get_dataset(std::string dataset_name, std::string csv_path, std::string save_path, int input_size, int label_size)
{
    Dataset dataset;

    if (dataset_exists(save_path))
    {
        std::cout << ">> Dataset exists! Loading dataset." << std::endl;
        dataset = load_dataset(save_path);
    }
    else
    {
        std::cout << ">> Dataset not found. Initializing dataset." << std::endl;
        dataset = load_csv(dataset_name, csv_path, input_size, label_size);
        save_dataset(save_path, dataset);
    }

    print_dataset(dataset);
    return dataset;
}
