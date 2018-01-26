#include "tensor2tt_lossless.h"
#include <unordered_map>
#include "xerus/misc/check.h"
#include "xerus/misc/internal.h"

namespace xerus {

TTTensor tensor2tt_lossless(Tensor b, int vpos) {
    const size_t d = b.degree();
    REQUIRE(d >= 2, "Invalid Tensor");
    vpos %= d;
    if (vpos < 0) {
        vpos += d;
    }
    
    TTTensor u(d);
    const auto &data = b.get_sparse_data();
    std::unordered_map<size_t, std::vector<std::pair<size_t, value_t>>> submats;
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

