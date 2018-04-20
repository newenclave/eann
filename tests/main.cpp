#include <iostream>

#include "eann/network.h"

static int random_value(int a, int b)
{
    std::random_device rd;
    std::default_random_engine re(rd());
    std::uniform_int_distribution<int> unif(a, b);
    return unif(re);
}

int main()
{

    eann::network<double> net({2, 3, 1});
    net.init_training();

    for (int i = 0; i < 10000; ++i)
    {
        auto x = random_value(0,1);
        auto y = random_value(0, 1);
        int res = x ^ y;
        net.forward_propagation({ (double)x, (double)y });
        net.backward_propagation({ (double)(res) });

        std::cout << std::round(net.results()[0]) << " " << res
            << " E: " << net.last_error()
            << "\n";
    }

    return 0;
}

