#include <iostream>
#include <fstream>
#include <memory>
#include <random>
#include <stdio.h>
#include "eann/network.h"

namespace {

    using float_type = double;

    const std::string paths[] = {
        "data0", "data1", "data2", "data3",
        "data4", "data5", "data6", "data7",
        "data8", "data9",
    };

    const std::string chars[] = {
        " ", "░", "▒", "▓", "█",
    };

    template <typename T=std::size_t>
    T random_value(T a, T b)
    {
        std::random_device rd;
        std::default_random_engine re(rd());
        std::uniform_int_distribution<T> unif(a, b);
        return unif(re);
    }

    constexpr std::size_t max_img = 1000;
    constexpr std::size_t img_size = 28*28;

    class data_source {
    public:

        data_source(const data_source &) = delete;


        data_source(std::string path)
            :path_(std::move(path))
            ,input_(img_size)
            ,target_(10)
        {}

        ~data_source()
        {
            for(auto &f: files_) {
                if(f) {
                    fclose(f);
                }
            }
        }

        void next()
        {
            auto num = random_value(0, 9);
            if(!files_[num]) {
                auto path = path_ + "/" + paths[num];
                files_[num] = fopen(path.c_str(), "rb");
            }
            auto img = random_value<std::size_t>(0, max_img);
            fseek(files_[num], img * img_size, SEEK_SET);
            fread(&data_[0], 1, img_size, files_[num]);
            current_ = num;
        }

        const std::vector<float_type> &input()
        {
            for(std::size_t i=0; i<input_.size(); ++i) {
                input_[i] = (static_cast<float_type>(data_[i]) / 255.0);
            };
            return input_;
        }

        const std::vector<float_type> &target()
        {
            std::vector<float_type> result(10);
            result[current_] = 1.0;
            target_.swap(result);
            return target_;
        }

        std::size_t current() const
        {
            return current_;
        }

        void show_current()
        {
            auto factor = 256 / 4;
            for(std::size_t r=0; r<28; ++r) {
                for(std::size_t c=0; c<28; ++c) {
                    std::cout << chars[data_[r * 28 + c] / factor];
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

    private:

        std::string path_;
        FILE *files_[10];
        std::uint8_t data_[28*28];
        std::vector<float_type> input_;
        std::vector<float_type> target_;
        std::size_t current_ = 0;
    };

    using network_type = eann::network<float_type>;

    std::pair<std::size_t, float_type>
    prediction(const std::vector<float_type> &result)
    {
        std::size_t max = 0;
        float_type  maxf = 0;
        std::size_t id = 0;
        for(auto r: result) {
            if(r > maxf) {
                maxf = r;
                max = id;
            }
            id++;
        }
        return std::make_pair(max, maxf);
    }

}


int main(int argc, char *argv[])
{
    if(argc < 2) {
        std::cout << "Path to data is missing\n";
        return 1;
    }

    network_type net {img_size, 128, 64, 10};
    net.init_training();

    data_source ds(argv[1]);

    std::size_t id = 0;
    while(true) {
        ++id;
        ds.next();
        net.forward_propagation(ds.input());
        net.backward_propagation(ds.target());
        if(id % 1000 == 0) {
            std::cout << ds.current() << " " << net.last_error() << "\n";
            auto pred = prediction(net.results());
            std::cout << pred.first << " " << pred.second << "\n";
            ds.show_current();
//            auto res = net.results();
//            for(auto r: res) {
//                std::cout << " " << r;
//            }
//            std::cout << "\n";
        }
    }

    ds.show_current();
    std::cout << ds.current() << "\n";

    return 0;
}
