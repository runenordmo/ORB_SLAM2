#include <iostream>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <array>
#include <vector>


struct cnnFeature{
    int x;
    int y;
    std::array<float,512> descriptor;
};

int main()
{
    std::string filename = "output/descriptors.dat";
    std::vector<cnnFeature> vFeatures;

    // Read from file
    std::ifstream f{filename, std::ios::binary};
    if (!f) { throw std::runtime_error{std::strerror(errno)}; }

    while( !f.eof() )
    {
        cnnFeature feat;

        // Read (x,y)-coordinate of feature
        f.read(reinterpret_cast<char*>(&feat.x), sizeof(std::int32_t));
        f.read(reinterpret_cast<char*>(&feat.y), sizeof(std::int32_t));

        bool last_feature_in_frame = feat.x==-1 || feat.y==-1;
        if( last_feature_in_frame ){
            std::cout << "Read all features for this frame!" << std::endl;
            break;
        }

        // Read descriptor of feature
        for (auto i = 0; i < feat.descriptor.size(); ++i) {
            float t;
            f.read(reinterpret_cast<char *>(&t), sizeof(t));
            if (!f) { throw std::runtime_error{std::strerror(errno)}; }
            feat.descriptor[i] = t;
        }

        // Save feature in vector of features
        vFeatures.push_back(feat);
    }

    for(auto feat : vFeatures) {
        std::cout << "("  << feat.x << ", " << feat.y << ")" << std::endl;
        for (auto descVal : feat.descriptor) {
            std::cout << descVal << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}