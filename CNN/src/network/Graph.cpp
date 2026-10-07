#include <cnn/network/Graph.hpp>

namespace cnn{
void LayerGraph::add_nodes(const std::vector<std::string>& names, const std::vector<cnn::Layer>& layers) {
    if (names.size() != layers.size())
        throw std::invalid_argument("Names and layers size mismatch");

    for (std::size_t i = 0; i < names.size(); ++i) {
        const std::string name = (names.empty()) ? utils::random_str() : names[i];

        if (nodes.count(name))
            throw std::invalid_argument("Each node must only present once in the graph");
        
        nodes[name] = layers[i];
    }  
}

void LayerGraph::add_edges(const std::vector<std::string>& from, const std::vector<std::string>& to) {
    if (from.size() != to.size())
        throw std::invalid_argument("The size of source and destination layers mismatch");

    for (std::size_t i = 0; i < from.size(); ++i) {
        if (!nodes.count(from[i]) || !nodes.count(to[i]))
            throw std::invalid_argument("All node must be present in the nodes data");

        if (edges.count(from[i]) && !edges[from[i]].empty()) {
            if (std::count(edges[from[i]].begin(), edges[from[i]].end(), from[i]))
                throw std::invalid_argument("A node is not allowed to connect to itself");
        }
    }
}

}
