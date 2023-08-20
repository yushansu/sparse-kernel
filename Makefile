UPD_NB ?= 80
FWD_KB ?= 128
BWD_CB ?= 128
JIT_NB ?= 16
JIT_nb ?= 16
JIT_CB ?= 128
JIT_KB ?= 128
JIT_UPD_NB ?= 16
JIT_UPD_nb ?= 16
JIT_UPD_CB ?= 32
JIT_UPD_KB ?= 32

EXEC = test
SRC = $(wildcard *.cc)
OBJS := $(addsuffix .o, $(basename $(SRC)))

CXX = icc 
CXXFLAGS = -g -O3 -m64 -Wall -fopenmp -std=c++11 -xCOMMON-AVX512
CXXFLAGS += -mkl=parallel
CXXFLAGS += -DUPD_NB_=${UPD_NB} -DFWD_KB_=${FWD_KB} -DBWD_CB_=${BWD_CB} \
            -DJIT_NB_=${JIT_NB} -DJIT_nb_=${JIT_nb} -DJIT_CB_=${JIT_CB} -DJIT_KB_=${JIT_KB} \
			-DJIT_UPD_NB_=${JIT_UPD_NB} -DJIT_UPD_nb_=${JIT_UPD_nb} -DJIT_UPD_CB_=${JIT_UPD_CB} -DJIT_UPD_KB_=${JIT_UPD_KB}
CXXFLAGS += -DLIBXSMM_OPENMP_SIMD -I/data/nfs_home//libxsmm/include
LIBS += -liomp5 -lpthread -lm -ldl -lnuma
LIBS += /data/nfs_home//libxsmm/lib/libxsmm.a \
        -Wl,--as-needed /data/nfs_home//libxsmm/lib/libxsmmnoblas.a -Wl,--no-as-needed \
		-Wl,--as-needed /data/nfs_home//libxsmm/lib/libxsmmext.a -Wl,--no-as-needed \
		-Wl,--gc-sections -Wl,-z,relro,-z,now -Wl,--export-dynamic \
		-Wl,--as-needed -lm -lrt -ldl -Wl,--no-as-needed \
		-Wl,--as-needed -lstdc++ -Wl,--no-as-needed

all: ${EXEC} 

$(EXEC): $(OBJS) Makefile
	$(CXX) ${CXXFLAGS} ${LDFLAGS} $(OBJS) -o $(EXEC) ${LIBS}

%.o : %.cc Makefile
	$(CXX) ${CXXFLAGS} ${INC} -c $< -o $@ 

clean:
	@rm -f main.o test
#	@rm -f *.o *.scsoa ${EXEC}
