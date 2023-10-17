#include "readjust_positions.hpp"


int main() {
    std::vector<double> axis = {1, 0, 0};
    std::vector<double> v = {0.879722, 0.0206041, -0.125461, -0.219793, 0.0574036, -0.244866, 0.105483, -0.243887, 0.0125734, -0.131068, 0.102061};
    for (auto &a: v)
        std::cout << a << " ";
    std::cout << std::endl;

    auto tmp1 = to_hyperspherical_coordinates(v);
    for (auto &a: tmp1)
        std::cout << a << " ";
    std::cout << std::endl;

    auto tmp2 = to_euclidean_coordinates(tmp1);
    for (auto &a: tmp2)
        std::cout << a << " ";
    std::cout << std::endl;

}
