#include <cnn/layer/Layer.hpp>

namespace cnn {
Layer::Layer(std::size_t in_channels=1, std::size_t out_channels=1,
             std::size_t in_data_side, std::size_t out_data_side) 
: in_channels(in_channels), out_channels(out_channels),
  in_data_side(in_data_side), out_data_side(out_data_side) {
    out_data.assign(out_data_side * out_data_side * out_channels, 0.0);
    deltas.assign(out_data_side * out_data_side * out_channels, 0.0);
    prev_dldas.assign(in_data_side * in_data_side * in_channels, 0.0);
}

void Layer::backward_residue_deltas() {
    for (Layer* source : from_residues) {
        auto& source_deltas = source->get_deltas();
        
        if (source_deltas.size() != out_data_side * out_data_side * out_channels)
            throw std::runtime_error("Source deltas size mistmatch");

        if (source->get_layer_type() == cnn::POOL) {

            #pragma omp parallel for schedule(static)
            for (std::size_t d = 0; d < out_data_side * out_data_side * out_channels; ++d)
                source_deltas[d] += deltas[d];

            continue;
        }

        cnn::ConvolutionLayer* source_layer = static_cast<cnn::ConvolutionLayer*>(source);
        const auto& source_pre = source_layer->get_pre_activated_out_data();

        switch (source_layer->get_activation_type()) {
            case dnn::RELU:

                #pragma omp parallel for schedule(static)
                for (std::size_t d = 0; d < out_data_side * out_data_side * out_channels; ++d)
                    source_deltas[d] += (source_pre[d] > 0.0) ? deltas[d] : 0.0;
                break;

            default:
                throw std::runtime_error("Unsupported activation function in residue layer");
        }

    }
}

OperationLayer::OperationLayer(std::size_t in_channels=1, std::size_t out_channels=1,
              std::size_t stride=1, std::size_t padding=0,
              std::size_t in_data_side, std::size_t out_data_side)
: Layer(in_channels, out_channels, in_data_side, out_data_side),
  stride(stride), padding(padding) {
    if (stride <= 0)
        throw std::invalid_argument("The stride must be greater than zero");
}

}