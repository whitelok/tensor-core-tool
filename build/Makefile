TARGET  	= CUDNN_TEST
CC 			= g++

CODE_DIR 	= ../
LIB_DIR		= ../lib

SRC_DIR 	= $(CODE_DIR)/src
INC_DIR 	= $(CODE_DIR)/inc

CUDA_DIR = /usr/local/cuda-10.0/
CUDA_INC_DIR = $(CUDA_DIR)/include
CUDA_LIB_DIR = $(CUDA_DIR)/lib64

vpath  		%.cpp $(SRC_DIR)
vpath  		%.cu  $(SRC_DIR)
vpath		%.h	  $(INC_DIR)

IFLAGS      = -I$(INC_DIR)\
	          -I$(CUDA_INC_DIR)

CFLAGS  	= $(IFLAGS)  -fPIC -O3 -Wall  $(FLAGS)
LDFLAGS     =  -L$(CUDA_LIB_DIR) -lcudart -lcudnn 

SRCS 		= $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS     = $(wildcard $(SRC_DIR)/*.cu)
INCS		= $(wildcard $(INC_DIR)/*.h)

OBJS		= $(notdir $(patsubst %.cpp, %.o, $(SRCS)))
OBJS		+= $(notdir $(patsubst %.cu, %.o, $(CU_SRCS)))

$(TARGET):	$(OBJS)
	$(CC) -o $(TARGET) $^ $(LDFLAGS)

%.o: %.c $(INCS)
	$(CC) -c $< $(CFLAGS)

clean:
	-rm $(TARGET)
	-rm $(OBJS)
