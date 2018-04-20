#pragma once

#include <random>
#include <cmath>
#include <cassert>

#include "layer.h"
#include "matrix.h"
#include "traits/activate_tanh.h"

namespace eann {

	template <typename DataType, 
	          typename CalcTrait = traits::activate_tanh<DataType> 
	>
	class network {
	public:
		using topology_type = std::vector<std::size_t>;
		using value_type = DataType;
		using layer_type = layer<value_type>;
		using layer_gradient_type = std::vector<value_type>;
		using neuron_type = neuron<value_type>;
		using matrix_type = matrix<value_type>;
		using calcs = CalcTrait;

		network(topology_type topology)
			:layers_(topology.size())
			,weights_(topology.size() - 1)
		{
			for (std::size_t t = 0; t < topology.size(); ++t) {
				layers_[t].resize(topology[t] + 1);
				layers_[t].back() = calcs::bias();
				if (t > 0) {
					weights_[t - 1] = matrix_type(topology[t - 1] + 1, topology[t]);
					weights_[t - 1].for_each([](value_type &val) {
						val = random_value(0.0, 1.0);
					});
				}
			}
		}

		void init_training()
		{
			for (auto &l : layers_) {
				gradients_.emplace_back(layer_gradient_type(l.size()));
			}

			for (auto &wmatrix : weights_) {
				deltas_.emplace_back(matrix_type(wmatrix.dimention()));
			}
		}

		void forward_propagation(const std::vector<value_type> &input)
		{
			assert(input.size() == (layers_.front().size() - 1));
			
			auto &input_layer = layers_.front();

			for (std::size_t i = 0; i < input.size(); ++i) {
				input_layer[i] = input[i];
			}

			for (std::size_t lid = 1; lid < layers_.size(); ++lid) {
				for (std::size_t nid = 0; nid<layers_[lid].size() - 1; ++nid) {
					neuron_forward(lid, nid);
				}
			}
		}

		void backward_propagation(const std::vector<value_type> &targets)
		{
			assert(targets.size() == (layers_.back().size() - 1));
			assert(layers_.size() == gradients_.size());
			assert(weights_.size() == deltas_.size());

			auto& output = layers_.back();
			auto& output_gradients = gradients_.back();

			// calc an error
			last_error_ = 0.0;
			for (std::size_t i = 0; i < targets.size(); ++i) {
				value_type delta = targets[i] - output[i];
				last_error_ += (delta * delta);
			}
			last_error_ /= static_cast<value_type>(targets.size());
			last_error_ = std::sqrt(last_error_);
			
			// calc output gradient
			for (std::size_t i = 0; i < targets.size(); ++i) {
				value_type delta = targets[i] - output[i];
				output_gradients[i] = delta * calcs::derive(output[i]);
			}
			
			// calc hidden layers gradients
			for (std::size_t lid = layers_.size() - 2; lid > 0; --lid) {
				auto &current = layers_[lid];
				for (std::size_t nid = 0; nid < current.size() - 1; ++nid) {
					neuron_calc_hidden_gradient(lid, nid);
				}
			}

			// updates weights
			for (std::size_t lid = layers_.size() - 1; lid > 0; --lid) {
				auto &current = layers_[lid];
				for (std::size_t nid = 0; nid < current.size() - 1; ++nid) {
					neuron_update_weights(lid, nid);
				}
			}
		}

		std::vector<value_type> results() const
		{
			std::vector<value_type> res;
			auto &output = layers_.back();
			for (auto &val : output) {
				res.emplace_back(val);
			}
			res.pop_back();
			return res;
		}

		value_type last_error() const
		{
			return last_error_;
		}

	private:
		
		void neuron_forward(std::size_t lid, std::size_t nid)
		{
			auto &prev_layer = layers_[lid - 1];
			auto &prev_weights = weights_[lid - 1];
			
			value_type sum = 0.0;
			for (std::size_t i = 0; i < prev_layer.size(); ++i) {
				sum += (prev_layer[i] * prev_weights[i][nid]);
			}
			sum /= static_cast<value_type>(prev_layer.size());
			layers_[lid][nid] = calcs::activate(sum);
		}

		void neuron_calc_hidden_gradient(std::size_t lid, std::size_t nid)
		{
			auto &current = layers_[lid];
			auto &current_weights = weights_[lid];
			auto &next = gradients_[lid + 1];

			// calc DOW 
			value_type dow_sum = 0.0;

			for (std::size_t i = 0; i < next.size() - 1; ++i) {
				dow_sum += (current_weights[nid][i] * next[i]);
			}
			gradients_[lid][nid] = dow_sum * calcs::derive(current[nid]);
		}

		void neuron_update_weights(std::size_t lid, std::size_t nid)
		{
			auto &current_gradient = gradients_[lid];
			auto &prev = layers_[lid - 1];
			auto &prev_delta = deltas_[lid - 1];
			auto &prev_weights = weights_[lid - 1];

			for (std::size_t i = 0; i < prev.size(); ++i) {
				value_type old_delta = prev_delta[i][nid];
				value_type new_delta =
					calcs::eta() * current_gradient[nid] * prev[i]
					+
					calcs::alfa() * old_delta;

				prev_weights[i][nid] += new_delta;
				prev_delta[i][nid] = new_delta;
			}
		}

		static value_type random_value(value_type a, value_type b)
		{
			std::random_device rd;
			std::default_random_engine re(rd());
			std::uniform_real_distribution<value_type> unif(a, b);
			return unif(re);
		}

	private:
		std::vector<layer_type> layers_;
		std::vector<matrix_type> weights_;

		std::vector<layer_gradient_type> gradients_;
		std::vector<matrix_type> deltas_;
		value_type last_error_ = 0.0;
	};
}
