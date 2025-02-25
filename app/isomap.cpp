// isomap.cpp
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <queue>
#include <limits>

namespace py = pybind11;
using namespace torch;
using namespace std;

// Calcular distancias euclidianas por pares
torch::Tensor pairwise_distances(torch::Tensor X) {
    int n = X.size(0);
    auto D = torch::zeros({n, n});
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float dist = torch::norm(X[i] - X[j]).item<float>();
            D[i][j] = D[j][i] = dist;
        }
    }
    return D;
}

// Calcular los paths mas cortos usando el algoritmo Dijkstra
torch::Tensor shortest_paths(torch::Tensor D, int k) {
    int n = D.size(0);
    auto G = torch::full({n, n}, std::numeric_limits<float>::infinity());
    for (int i = 0; i < n; i++) {
        std::vector<std::pair<float, int>> neighbors;
        for (int j = 0; j < n; j++) {
            if (i != j) neighbors.push_back({D[i][j].item<float>(), j});
        }
        std::sort(neighbors.begin(), neighbors.end());
        for (int m = 0; m < k && m < (int)neighbors.size(); m++) {
            G[i][neighbors[m].second] = neighbors[m].first;
        }
    }
    return G;
}

// Calcular la proyeccion Isomap
tuple<torch::Tensor, torch::Tensor> isomap(torch::Tensor X, int k, int d) {
    auto D = pairwise_distances(X);
    auto G = shortest_paths(D, k);
    auto G_floyd = torch::from_blob(G.data_ptr<float>(), {G.size(0), G.size(1)}, torch::kFloat);
    for (int k = 0; k < G.size(0); k++) {
        for (int i = 0; i < G.size(0); i++) {
            for (int j = 0; j < G.size(0); j++) {
                G_floyd[i][j] = std::min(G_floyd[i][j].item<float>(), 
                                         G_floyd[i][k].item<float>() + G_floyd[k][j].item<float>());
            }
        }
    }

    // Reemplazar infinitos con el máximo valor finito
    G_floyd = torch::where(G_floyd == std::numeric_limits<float>::infinity(), 
                           torch::max(G_floyd.masked_select(G_floyd < std::numeric_limits<float>::infinity())), 
                           G_floyd);

    auto H = torch::eye(G.size(0)) - torch::ones({G.size(0), G.size(0)}) / G.size(0);
    auto B = -0.5 * H.matmul(G_floyd).matmul(H);
    auto svd = torch::svd(B);
    auto U = std::get<0>(svd);
    auto S = std::get<1>(svd);
    auto Y = U.index({torch::indexing::Slice(), torch::indexing::Slice(0, d)}) * 
             torch::sqrt(S.index({torch::indexing::Slice(0, d)})).unsqueeze(0);
    return make_tuple(Y, G_floyd);
}

PYBIND11_MODULE(isomap_module, m) {
    m.def("isomap", &isomap, "Isomap dimensionality reduction", py::arg("X"), py::arg("k"), py::arg("d"));
}
