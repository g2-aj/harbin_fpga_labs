# Copyright (C) 2013-2016 Altera Corporation, San Jose, California, USA. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
# whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# 
# This agreement shall be governed in all respects by the laws of the State of California and
# by the laws of the United States of America.
# This is a GNU Makefile.

# You must configure ALTERAOCLSDKROOT to point the root directory of the Altera SDK for OpenCL
# software installation.
# See http://www.altera.com/literature/hb/opencl-sdk/aocl_getting_started.pdf 
# for more information on installing and configuring the Altera SDK for OpenCL.
VERBOSE:=1
EMU:=1
ifeq ($(VERBOSE),1)
ECHO := 
else
ECHO := @
endif

# Where is the Altera SDK for OpenCL software?
ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation.)
endif

# OpenCL compile and link flags.
ifeq ($(EMU), 1)
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := -Wl,--no-as-needed $(shell aocl link-config)
else 
AOCL_COMPILE_CONFIG := $(shell aocl compile-config --arm )
AOCL_LINK_CONFIG := -Wl,--no-as-needed $(shell aocl link-config --arm)
endif
    

# Compilation flags
ifeq ($(DEBUG),1)
CXXFLAGS += -g
else
CXXFLAGS += -O2
endif

# Compiler
CROSS-COMPILE = arm-linux-gnueabihf-
CXX := g++

TARGET_EMU := lab3-emu
TARGET_HW := lab2-hw
TARGET_DIR := bin

# Directories
INC_DIRS := ../../common/inc
LIB_DIRS := 

# Files
INCS := $(wildcard *.h)
SRCS := $(wildcard *.cpp ../../common/src/AOCLUtils/*.cpp)
LIBS := rt

# Make it all!
emu : $(TARGET_DIR)/$(TARGET_EMU)
hw : $(TARGET_DIR)/$(TARGET_HW)

# Host executable target.
$(TARGET_DIR)/$(TARGET_EMU) : Makefile $(SRCS) $(INCS) $(TARGET_DIR)
	$(ECHO)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
			$(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) \
			$(foreach D,$(LIB_DIRS),-L$D) \
			$(foreach L,$(LIBS),-l$L) \
			-o $(TARGET_DIR)/$(TARGET_EMU)

$(TARGET_DIR)/$(TARGET_HW) : Makefile $(SRCS) $(INCS) $(TARGET_DIR)
	$(ECHO)$(CROSS-COMPILE)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
			$(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) \
			$(foreach D,$(LIB_DIRS),-L$D) \
			$(foreach L,$(LIBS),-l$L) \
			-o $(TARGET_DIR)/$(TARGET_HW)

$(TARGET_DIR) :
	$(ECHO)mkdir $(TARGET_DIR)

run-emu: $(TARGET_DIR)/$(TARGET_EMU)
	CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 $(TARGET_DIR)/$(TARGET_EMU) bin/conv_kernel \
		mnist_test_img_0.pgm

run-hw: $(TARGET_DIR)/$(TARGET_HW)
	$(TARGET_DIR)/$(TARGET_HW) bin/conv_kernel \
		mnist_test_img_0.pgm

# Standard make targets
clean :
	$(ECHO)rm -rf $(TARGET_DIR)
	$(ECHO)rm -f fpga_output.pgm

print :
	@echo "Sources : $(SRCS)"
	@echo "Includes : $(INC_DIRS)"
.PHONY : all clean
