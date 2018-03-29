#pragma once

#include "xerus/ttNetwork.h"

namespace xerus {

void printMatrix(Tensor a);

TTTensor tensor2tt_lossless(Tensor b, int vpos = -1);

}

