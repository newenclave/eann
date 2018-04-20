#include <cmath>

namespace eann {

	namespace traits {

		template <typename DataType>
		struct activate_tanh {
			
			static DataType bias()
			{
				return 1.0;
			}

			static DataType activate(DataType data)
			{
				return tanh(data);
			}

			static DataType derive(DataType data)
			{
				return 1.0 - data * data;
			}

			static DataType eta()
			{
				return 0.0315;
			}

			static DataType alfa()
			{
				return 0.5;
			}

		};
	}
}