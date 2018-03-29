#include "tensor2tt_lossless.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <limits>
#include "xerus/misc/check.h"
#include "xerus/misc/internal.h"

using namespace std;

namespace xerus {

void printMatrix(Tensor a) {
    const size_t d = a.degree();
    REQUIRE(d == 2, "unimplement");
    const auto &n = a.dimensions;
    const size_t nrows = n.at(0);
    const size_t ncols = n.at(1);
    cout << nrows << " " << ncols << endl;
    for (size_t i = 0; i < nrows; ++i) {
        for (size_t j = 0; j < ncols; ++j) {
            cout << a[{i, j}] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

auto depar01(Tensor a) {
    const size_t d = a.degree();
    REQUIRE(d == 2, "Input of depar01 must be a matrix");
    const size_t nrows = a.dimensions.at(0);
    const size_t ncols = a.dimensions.at(1);
    
    vector<size_t> a_col_nze(ncols, numeric_limits<size_t>::max());
    a.use_sparse_representation();
    const auto &data = a.get_sparse_data();
    for (const auto &entry : data) {
        if (entry.second != 1) {
            cout << entry.second << endl;
        }
        auto indices = Tensor::position_to_multiIndex(entry.first, a.dimensions);
        size_t index_row = indices.at(0);
        size_t index_col = indices.at(1);
        a_col_nze.at(index_col) = index_row;
    }
    
    vector<size_t> hashmap(nrows, numeric_limits<size_t>::max());
    vector<vector<size_t>> b_indices;
    vector<vector<size_t>> t_indices;
    size_t cnt = 0;
    for (size_t i = 0; i < ncols; ++i) {
        const size_t &index_row = a_col_nze.at(i);
        if (index_row >= nrows) {
            continue;
        }
        size_t &newid = hashmap.at(index_row);
        if (newid >= ncols) {
            b_indices.push_back({index_row, cnt});
            newid = cnt++;
        }
        t_indices.push_back({newid, i});
    }
    
    vector<size_t> b_size{nrows, cnt};
    vector<size_t> t_size{cnt, ncols};
    map<size_t, value_t> b_data;
    map<size_t, value_t> t_data;
    for (const auto &entry : b_indices) {
        size_t pos = Tensor::multiIndex_to_position(entry, b_size);
        b_data.try_emplace(pos, 1);
    }
    for (const auto &entry : t_indices) {
        size_t pos = Tensor::multiIndex_to_position(entry, t_size);
        t_data.try_emplace(pos, 1);
    }
    
    auto b = Tensor(b_size, Tensor::Representation::Sparse, Tensor::Initialisation::None);
    b.get_unsanitized_sparse_data() = move(b_data);
    auto t = Tensor(t_size, Tensor::Representation::Sparse, Tensor::Initialisation::None);
    t.get_unsanitized_sparse_data() = move(t_data);
    
    return make_tuple(move(b), move(t));
}

void parrounding(TTTensor &a, size_t vpos) {
    const size_t d = a.degree();
    for (size_t i = 0; i < vpos; ++i) {
        auto &comp = a.component(i);
        const auto n = comp.dimensions;
        comp.reinterpret_dimensions({n.at(0) * n.at(1), n.at(2)});
        auto [b, t] = depar01(move(comp));
        b.reinterpret_dimensions({n.at(0), n.at(1), b.dimensions.back()});
        a.set_component(i, move(b));
        auto &succ = a.component(i+1);
        contract(t, t, succ, 1);
        a.set_component(i+1, move(t));
    }
    for (size_t i = d-1; i > vpos; --i) {
        auto &comp = a.component(i);
        const auto n = comp.dimensions;
        comp.reinterpret_dimensions({n.at(0), n.at(1) * n.at(2)});
        reshuffle(comp, comp, {1, 0});
        auto [b, t] = depar01(move(comp));
        reshuffle(b, b, {1, 0});
        b.reinterpret_dimensions({b.dimensions.front(), n.at(1), n.at(2)});
        a.set_component(i, move(b));
        auto &pre = a.component(i-1);
        contract(t, pre, false, t, true, 1);
        a.set_component(i-1, move(t));
    }
}

TTTensor tensor2tt_lossless(Tensor a, int vpos) {
    const size_t d = a.degree();
    REQUIRE(d >= 2, "Invalid Tensor");
    vpos %= d;
    if (vpos < 0) {
        vpos += d;
    }
    
    TTTensor u(d);
    a.use_sparse_representation();
    const auto &data = a.get_sparse_data();
    unordered_map<size_t, vector<pair<size_t, value_t>>> submats;
    const auto &n = a.dimensions;
    size_t pn = 1;
    for (size_t i = vpos + 1; i < d; ++i) {
        pn *= n.at(i);
    }
    for (const auto &entry : data) {
        const size_t in_pos = entry.first / pn % n.at(vpos);
        const size_t out_pos = entry.first - in_pos * pn;
        submats[out_pos].emplace_back(in_pos, entry.second);
    }
    
    const size_t r = submats.size();
    u.set_component(0, Tensor({1, n.front(), r}));
    for (size_t i = 1; i < d - 1; ++i) {
        u.set_component(i, Tensor({r, n.at(i), r}));
    }
    u.set_component(d - 1, Tensor({r, n.back(), 1}));
    
    size_t index = 0;
    for (const auto &submat : submats) {
        const size_t out_pos = submat.first;
        const auto md = Tensor::position_to_multiIndex(out_pos, n);
        for (size_t i = 0; i < d; ++i) {
            size_t nleft = i == 0 ? 0 : index;
            size_t nright = i == d - 1 ? 0 : index;
            if (i != static_cast<size_t>(vpos)) {
                u.component(i).operator[]({nleft, md.at(i), nright}) = 1;
            }
            else {
                for (const auto &entry : submat.second) {
                    const size_t in_pos = entry.first;
                    const value_t v = entry.second;
                    u.component(i).operator[]({nleft, in_pos, nright}) = v;
                }
            }
        }
        ++index;
    }
    
    parrounding(u, vpos);
    
    return u;
}

}

