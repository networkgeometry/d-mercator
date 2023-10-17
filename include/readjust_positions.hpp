#ifndef MERCATOR_READJUST_POSITIONS
#define MERCATOR_READJUST_POSITIONS

#include <iostream>
#include <cmath>
#include "Eigen/Dense"

const double PI = 3.141592653589793238462643383279502884197;

Eigen::MatrixXd compute_rotation_matrix(std::vector<double> &axis, std::vector<double> &v) {
    const int size = axis.size();
    Eigen::VectorXd vec_axis = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(axis.data(), size);
    Eigen::VectorXd vec_v = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(v.data(), size);

    const auto sum = vec_axis + vec_v;
    const auto sum_T = sum.transpose();
    const auto rotation_matrix = 2 * (sum * sum_T) / (sum_T * sum) -  Eigen::MatrixXd::Identity(size, size);
    return rotation_matrix;
}

std::vector<double> rotate_vector(Eigen::MatrixXd &rotation_matrix, std::vector<double> &v) {
    Eigen::VectorXd vec_v = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(v.data(), v.size());
    Eigen::VectorXd rotated_vec = rotation_matrix * vec_v;
    std::vector<double> new_vec;
    new_vec.resize(rotated_vec.size());
    Eigen::VectorXd::Map(&new_vec[0], rotated_vec.size()) = rotated_vec;
    return new_vec;
}

std::vector<double> invert_rotation(Eigen::MatrixXd &rotation_matrix, std::vector<double> &v) {
    Eigen::MatrixXd inversed_rotation_matrix = rotation_matrix.inverse();
    return rotate_vector(inversed_rotation_matrix, v);
}

std::vector<double> to_hyperspherical_coordinates(const std::vector<double>& v) {
    // Assumption: ||v|| = 1 -> r = 1
    // Return all angles: phi_1, phi_2, ..., phi_{N-2}, theta
    // where phi_i = [0, pi], theta = [0, 2pi)
    int size = v.size();
    std::vector<double> angles;
    for (int i=0; i<size-2; ++i) {
        double bottom = 0;
        for (int j=i; j<size; ++j)
            bottom += v[j] * v[j];
        bottom = std::sqrt(bottom);
        angles.push_back(std::acos(v[i] / bottom));
    }
    double theta = std::acos(v[size - 2] / std::sqrt(v[size - 1] * v[size - 1] + v[size - 2] * v[size - 2]));
    if (v[size - 1] < 0)
        theta = 2 * PI - theta;
    if (theta != theta) // if theta is nan
        theta = 2 * PI;
    angles.push_back(theta);
    return angles;
}

std::vector<double> to_euclidean_coordinates(const std::vector<double>& v) {
    std::vector<double> positions;
    double val;
    for (int i=0; i<v.size(); ++i) {
        val = std::cos(v[i]);
        for (int j=0; j<i; ++j)
            val *= std::sin(v[j]);
        positions.push_back(val);

        if (i == v.size() - 1) { // last position
            val = std::sin(v[i]);
            for (int j=0; j<i; ++j)
                val *= std::sin(v[j]);
            positions.push_back(val);
        }
    }
    return positions;
}


#endif // MERCATOR_READJUST_POSITIONS
