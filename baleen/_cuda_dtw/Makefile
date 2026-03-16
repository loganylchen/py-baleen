# Makefile
CC = nvcc
CFLAGS = -std=c++11 -O3 -Xcompiler -fPIC
LDFLAGS = -lcudart

# 编译为动态库
libopendba_dtw.so: dtw_c_api.cpp dtw.hpp cuda_utils.hpp
	$(CC) $(CFLAGS) -shared -o $@ $< $(LDFLAGS)

# 清理编译产物
clean:
	rm -f libopendba_dtw.so