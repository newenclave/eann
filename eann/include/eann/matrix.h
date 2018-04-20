#pragma once
#include <cstdint>
#include <algorithm>

namespace eann {

    template <typename DataType>
    class matrix {
    public:

        struct dimention_type {
            dimention_type(std::size_t r, std::size_t c)
                :row(r)
                ,col(c)
            {}
            dimention_type() = default;

            std::size_t row = 0;
            std::size_t col = 0;
        };

        matrix() = default;
        matrix(matrix &&) = default;
        matrix & operator = (matrix &&) = default;

        matrix(std::size_t rows, std::size_t cols)
            :dimention_(rows, cols)
            ,storage_(rows * cols)
        { }

        matrix(dimention_type dim)
            :dimention_(dim)
            ,storage_(dim.row * dim.col)
        { }

        using value_type = DataType;
        using store_type = std::vector<value_type>;

        value_type *operator [] (std::size_t row)
        {
            return &storage_[row * dimention_.col];
        }

        const value_type *operator [] (std::size_t row) const
        {
            return &storage_[row * dimention_.col];
        }

        std::size_t rows() const
        {
            return dimention_.row;
        }

        std::size_t cols() const
        {
            return dimention_.col;
        }

        const dimention_type &dimention() const
        {
            return dimention_;
        }

        template <typename Call>
        void for_each(Call init)
        {
            for (auto &element : storage_) {
                init(element);
            }
        }

    private:
        dimention_type dimention_;
        store_type storage_;
    };
}
