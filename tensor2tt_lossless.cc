#include "tensor2tt_lossless.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "xerus/misc/check.h"
#include "xerus/misc/internal.h"

using namespace std;

namespace xerus {

auto depar01(Tensor a) {
    const size_t d = a.degree();
    REQUIRE(d == 2, "Input of depar01 must be a matrix");
    const size_t nrows = a.dimensions.at(0);
    const size_t ncols = a.dimensions.at(1);
    
    vector<u32string> a_col_nze(ncols);
    const auto &data = a.get_sparse_data();
    for (const auto &entry : data) {
        auto indices = Tensor::position_to_multiIndex(entry.first, a.dimensions);
        size_t index_row = indices.at(0);
        size_t index_col = indices.at(1);
        a_col_nze.at(index_col).push_back(static_cast<char32_t>(index_row));
    }
    
    unordered_map<u32string, size_t> hashmap;
    vector<vector<size_t>> b_indices;
    vector<vector<size_t>> t_indices;
    vector<int> a_col_newid(ncols);
    size_t cnt = 0;
    for (size_t i = 0; i < ncols; ++i) {
        if (a_col_nze.at(i).empty()) {
            continue;
        }
        const auto &sequence = a_col_nze.at(i);
        const auto ret = hashmap.try_emplace(sequence, cnt);
        const size_t newid = ret.first->second;
        a_col_newid.at(i) = newid;
        if (ret.second) {
            for (auto ch : sequence) {
                size_t index_row = static_cast<size_t>(ch);
                b_indices.push_back({index_row, cnt});
            }
            ++cnt;
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
    
    auto b = make_unique<Tensor>(b_size, Tensor::Representation::Sparse, Tensor::Initialisation::None);
    b->get_unsanitized_sparse_data() = move(b_data);
    auto t = make_unique<Tensor>(t_size, Tensor::Representation::Sparse, Tensor::Initialisation::None);
    t->get_unsanitized_sparse_data() = move(t_data);
    
    return make_tuple(move(b), move(t));
}

TTTensor tensor2tt_lossless(Tensor b, int vpos) {
    const size_t d = b.degree();
    REQUIRE(d >= 2, "Invalid Tensor");
    vpos %= d;
    if (vpos < 0) {
        vpos += d;
    }
    
    TTTensor u(d);
    const auto &data = b.get_sparse_data();
    unordered_map<size_t, vector<pair<size_t, value_t>>> submats;
    const auto &n = b.dimensions;
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
    
    return u;
}

}

