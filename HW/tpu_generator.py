import numpy as np
import argparse
import random
import os
import struct

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])[2:]
    

def write_matrix(fd, m):

    # r, c = m.shape


    # calign = int((c+3)/4)*4

    # malign = np.zeros((r, calign), dtype=np.uint8)

    # malign[:, 0:c] = m
    # cptr = 0

    # fd.write("\n")
    # for cptr in range(0, calign, 4):
    #     for dr in range(r):
    #         for n in range(4):
    #             # t = f"{malign[dr, cptr+n]}"
    #             fd.write("{0:2x} ".format(malign[dr, cptr+n]))
    #         fd.write('\n')
    Am = m
    in_fd = fd
    rows, cols = Am.shape

    for i in range(rows):
        for j in range(cols):
            # Convert the float to its IEEE 754 single-precision representation
            # and then to hex.  Use 'f' for single-precision (32-bit).
            # print(Am[i, j])
            # hex_representation = struct.pack('d', Am[i, j]).hex()
            hex_representation = float_to_hex(Am[i, j])
            in_fd.write(hex_representation + " ")
        in_fd.write("\n")

    fd.write("\n")

    # return malign

def write_output_matrix(fd, m):
    # r, c = m.shape


    # calign = int((c+3)/4)*4

    # malign = np.zeros((r, calign), dtype=np.uint32)

    # malign[:, 0:c] = m
    # cptr = 0

    # fd.write("\n")
    # for cptr in range(0, calign, 4):
    #     for dr in range(r):
    #         for n in range(4):
    #             # t = f"{malign[dr, cptr+n]}"
    #             fd.write("{0:8x} ".format(malign[dr, cptr+n]))
    #         fd.write('\n')
            
    # fd.write("\n")

    Am = m
    in_fd = fd
    rows, cols = Am.shape

    for i in range(rows):
        for j in range(cols):
            # Convert the float to its IEEE 754 single-precision representation
            # and then to hex.  Use 'f' for single-precision (32-bit).
            # print(Am[i, j])
            # hex_representation = struct.pack('d', Am[i, j]).hex()
            hex_representation = float_to_hex(Am[i, j])
            in_fd.write(hex_representation + " ")
        in_fd.write("\n")

    fd.write("\n")

    # return malign


def write_readable(fd, m, desc=""):
    r, c = m.shape

    fd.write(desc)
    for dr in range(r):
        for dc in range(c):
            fd.write("{0:f} ".format( m[dr, dc]))
        fd.write("\n")
    fd.write("\n")

def write_config(fd, K, M, N):

    fd.write("{0:3x} {1:3x} {2:3x}\n".format( K, M, N))


def gen_one_case(i, in_fd=None, c_fd=None, all_one=False, mode=0, shape_range=(4, 255), val_range=(0, 255)):

    if in_fd == None:
        print("input file descriptor is null")
        exit(2)
    
    if c_fd == None:
        print("check file descriptor is null")
        exit(2)

    #* generate the matrix

    # if all_one:
    #     Am = np.ones((M, K), dtype=np.uint8)
    #     Bm = np.ones((K, N), dtype=np.uint8)
    # elif mode == 4:
    #     Am = np.random.randint(-128, high=127, size=(M, K), dtype=np.int8)
    #     Bm = np.random.randint(-128, high=127, size=(K, N), dtype=np.int8) 
    # else:
        # np.random.seed(0)
        # Am = np.random.randint(val_range[0], high=val_range[1], size=(M, K), dtype=np.uint8)
    M = 8
    K = 8
    N = 8
    np.random.seed(0)
    img = np.random.rand(M, K).astype(np.float32)

    # weight
    kernel = np.random.rand(3, 3).astype(np.float32)
    kernel_2 = np.random.rand(3, 3).astype(np.float32)
    kernel_3 = np.random.rand(3, 3).astype(np.float32)
    kernel_4 = np.random.rand(3, 3).astype(np.float32)
    kernel_5 = np.random.rand(3, 3).astype(np.float32)
    
    img_height = len(img)
    img_width = len(img[0])
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image manually
    padded_img = [[0] * (img_width + 2 * pad_width) for _ in range(img_height + 2 * pad_height)]
    for i in range(img_height):
        for j in range(img_width):
            padded_img[i + pad_height][j + pad_width] = img[i][j]

    output   = [[0] * img_width for _ in range(img_height)]
    output_2 = [[0] * img_width for _ in range(img_height)]
    output_3 = [[0] * img_width for _ in range(img_height)]
    output_4 = [[0] * img_width for _ in range(img_height)]
    output_5 = [[0] * img_width for _ in range(img_height)]

    for i in range(img_height):
        for j in range(img_width):
            sum_val = 0
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    sum_val += padded_img[i + ki][j + kj] * kernel[ki][kj]
            output[i][j] = sum_val
    # 2
    for i in range(img_height):
        for j in range(img_width):
            sum_val = 0
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    sum_val += padded_img[i + ki][j + kj] * kernel_2[ki][kj]
            output_2[i][j] = sum_val
    # 3
    for i in range(img_height):
        for j in range(img_width):
            sum_val = 0
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    sum_val += padded_img[i + ki][j + kj] * kernel_3[ki][kj]
            output_3[i][j] = sum_val
    # 4
    for i in range(img_height):
        for j in range(img_width):
            sum_val = 0
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    sum_val += padded_img[i + ki][j + kj] * kernel_4[ki][kj]
            output_4[i][j] = sum_val
    # 5
    for i in range(img_height):
        for j in range(img_width):
            sum_val = 0
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    sum_val += padded_img[i + ki][j + kj] * kernel_5[ki][kj]
            output_5[i][j] = sum_val

    output = np.array(output)
    output_2 = np.array(output_2)
    output_3 = np.array(output_3)
    output_4 = np.array(output_4)
    output_5 = np.array(output_5)

    write_config(in_fd, K, M, N)

    padded_img = np.array(padded_img)
    write_matrix(in_fd, padded_img)
    write_matrix(in_fd, kernel)
    write_matrix(in_fd, kernel_2)
    write_matrix(in_fd, kernel_3)
    write_matrix(in_fd, kernel_4)
    write_matrix(in_fd, kernel_5)
    write_output_matrix(in_fd, output)
    write_output_matrix(in_fd, output_2)
    write_output_matrix(in_fd, output_3)
    write_output_matrix(in_fd, output_4)
    write_output_matrix(in_fd, output_5)


    c_fd.write("----------------------------------------------\n")
    c_fd.write(f"                 Case {i}                    \n")
    c_fd.write("----------------------------------------------\n")
    c_fd.write(f"K: {K:3d} M: {M:3d} N:{N:3d}\n")
    write_readable(c_fd, padded_img, desc="A: \n")
    write_readable(c_fd, kernel, desc ="B: \n")
    write_readable(c_fd, kernel_2, desc ="B: \n")
    write_readable(c_fd, kernel_3, desc ="B: \n")
    write_readable(c_fd, kernel_4, desc ="B: \n")
    write_readable(c_fd, kernel_5, desc ="B: \n")
    write_readable(c_fd, output, desc="C: \n")
    write_readable(c_fd, output_2, desc="C: \n")
    write_readable(c_fd, output_3, desc="C: \n")
    write_readable(c_fd, output_4, desc="C: \n")
    write_readable(c_fd, output_5, desc="C: \n")


def main():


    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=int, required=True, default=0, help="The mode of the generated version")
    parser.add_argument('--ncases', type=int, required=True, default=1, help="The number of cases to be generated")
    parser.add_argument('--ones', action="store_true", help="To specify all content of matrix is one")
    parser.add_argument('--target_dir', type=str, required=True, help="The Output directory for testcases")



    args = parser.parse_args()


    mode = args.mode
    ncases = args.ncases
    all_one = True if args.ones else False
    target_dir = args.target_dir


    try:
        os.mkdir(target_dir, 0o755)
    except OSError as error:
        print(error)

    input_file = os.path.join(target_dir, "input.txt")
    leg_file   = os.path.join(target_dir, "check.txt")

    in_fd = open(input_file, "w")
    legible_fd = open(leg_file, "w")

    in_fd.write(f"{ncases:d}")

    # ncases = 1
    for n in range(ncases):
        gen_one_case(n, in_fd, legible_fd, all_one=all_one, mode=mode, shape_range=(4, 127), val_range=(0, 255))



if __name__ == "__main__":
    main()