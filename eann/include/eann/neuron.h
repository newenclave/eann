#pragma once
#include <cstdint>
#include <vector>

namespace eann {
	template <typename DataType>
	class neuron {
	public:
		using value_type = DataType;		
		neuron() = default;

		neuron(value_type val)
			:output_(val)
		{}

		value_type output() const
		{
			return output_;
		}

		operator value_type () const
		{
			return output_;
		}

	private:
		value_type output_ = 0.0;
	};


}

