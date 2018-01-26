CXX = g++-7
LINK.o = $(LINK.cc)
DEBUG ?= 0
CXXFLAGS = -std=c++14 -Wall -Wextra
ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG -ggdb
else
	CXXFLAGS += -O3 -DNDEBUG -funroll-loops -fopenmp -D_GLIBCXX_PARALLEL -march=native
endif
OUTPUT_OPTION = -MMD -MP -o $@
LDLIBS = -lxerus -lxerus_misc

SRC = $(wildcard *.cc)
OBJ = $(SRC:.cc=.o)
DEP = $(SRC:.cc=.d)

.PHONY: all clean

EXEC = test

all: $(EXEC)

test: test.o tensor2tt_lossless.o

-include $(DEP)

clean:
	-rm -f $(OBJ) $(DEP) $(EXEC)

