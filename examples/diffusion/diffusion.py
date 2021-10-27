import numpy as np
import torch

# combine write_layer combinations into functions

def laplacian(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)

    write_layer[buffer_size : -buffer_size, buffer_size : -buffer_size] = roll_up[buffer_size : -buffer_size, buffer_size : -buffer_size] + \
                                                                          roll_down[buffer_size : -buffer_size, buffer_size : -buffer_size] + \
                                                                          roll_left[buffer_size : -buffer_size, buffer_size : -buffer_size] + \
                                                                          roll_right[buffer_size : -buffer_size, buffer_size : -buffer_size] + \
                                                                          read_layer[buffer_size : -buffer_size, buffer_size : -buffer_size] - 4 * read_layer[buffer_size : -buffer_size, buffer_size : -buffer_size]



def laplacian_tlrb(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)

    write_layer[0, 0] = roll_up[0, 0 ] + roll_left[0, 0] - 2 * read_layer[0, 0 ]
    # print(roll_up[0 : buffer_size, 0 : buffer_size])
    # print(roll_left[0 : buffer_size, 0 : buffer_size])
    # print(read_layer[0 : buffer_size, 0 : buffer_size])
    # left center
    write_layer[1 : - 1, 0] = roll_down[1 : -1, 0] + roll_up[1 : -1, 0 ] + roll_left[1 : -1, 0] - 3 * read_layer[1:-1, 0 ]
    # left bottom
    write_layer[- 1, 0] = roll_down[-1, 0] + roll_left[-1, 0] - 2 * read_layer[-1, 0 ]

    # right top
    write_layer[0, -1] = roll_up[0, -1 ] + roll_right[0, -1] - 2 * read_layer[0, -1]
    # right center
    write_layer[1 : -1, -1] = roll_up[1:-1, -1] + roll_down[1:-1, -1] + roll_right[1:-1, -1] - 3 * read_layer[1:-1, -1]
    # right bottom
    write_layer[-1, -1] = roll_down[-1, -1 ] + roll_right[-1, -1] - 2 *  read_layer[-1, -1]

    # center top
    write_layer[0, 1 : -1] = roll_up[0, 1 : -1] +  roll_left[0, 1 : -1] + roll_right[0, 1 : -1] - 3 * read_layer[0, 1 : -1]
    # center bottom
    write_layer[-1, 1 : -1] = roll_down[-1, 1 : -1] +  roll_left[-1, 1 : -1] + roll_right[-1, 1 : -1] + read_layer[-1, 1 : -1]
    write_layer[1 : -1, 1 : -1] = roll_up[1 : -1, 1 : -1] + \
                                                            roll_down[1 : -1, 1 : -1] + \
                                                            roll_left[1 : -1, 1 : -1] + \
                                                            roll_right[1 : -1, 1 : -1] + \
                                                            - 4 * read_layer[1 : -1, 1 : -1] 

                                                            # read_layer[1 : -1, 1 : -1] - 4 * read_layer[1 : -1, 1 : -1] 


def laplacian_tlr(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)

    # left right top unbuffered
    # left top
    write_layer[0, 0] = roll_up[0, 0 ] + roll_left[0, 0] - 2 * read_layer[0, 0 ]
    # left center
    write_layer[1 : -buffer_size, 0] = roll_down[1 : -buffer_size, 0] + roll_up[1 : -buffer_size, 0 ] + roll_left[1 : -buffer_size, 0] - 3 * read_layer[1:-buffer_size, 0 ]
    # right top
    write_layer[0, -1] = roll_up[0, -1 ] + roll_right[0, -1] - 2 * read_layer[0, -1]
    # right center
    write_layer[1 : -buffer_size, -1] = roll_down[1 : -buffer_size, 0] + roll_up[1 : -buffer_size, 0 ] + roll_right[1 : -buffer_size, 0] - 3 * read_layer[1:-buffer_size, 0 ]
    # center top
    write_layer[0, 1 : -1] = roll_up[0, 1 : -1] +  roll_left[0, 1 : -1] + roll_right[0, 1 : -1] - 3 * read_layer[0, 1 : -1]

    write_layer[1 : -buffer_size, 1 : -1] = roll_up[1 : -buffer_size, 1 : -1] + \
                                                            roll_down[1 : -buffer_size, 1 : -1] + \
                                                            roll_left[1 : -buffer_size, 1 : -1] + \
                                                            roll_right[1 : -buffer_size, 1 : -1] - \
                                                            4 * read_layer[1 : -buffer_size, 1 : -1]

def laplacian_tlb(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)

    # everything but right
    # left top
    write_layer[0, 0] = roll_up[0, 0 ] + roll_left[0, 0]  - 2 * read_layer[0, 0 ]
    # left center
    write_layer[1 : - 1, 0] = roll_down[1 : -1, 0] + roll_up[1 : -1, 0 ] + roll_left[1 : -1, 0] - 3 * read_layer[1:-1, 0 ]
    # left bottom
    write_layer[-1, 0] = roll_down[-1, 0] + roll_left[-1, 0] - 2 * read_layer[-1, 0 ]
    # center top
    write_layer[0, 1 : -buffer_size] = roll_up[0, 1 : -buffer_size] +  roll_left[0, 1 : -buffer_size] + roll_right[0, 1 : -buffer_size] - 3 * read_layer[0, 1 : -buffer_size]
    # center bottom
    write_layer[-1, 1 : -buffer_size] = roll_down[-1, 1 : -buffer_size] +  roll_left[-1, 1 : -buffer_size] + roll_right[-1, 1 : -buffer_size] + read_layer[-1, 1 : -buffer_size]
    write_layer[1 : -1, 1 : -buffer_size] = roll_up[1 : -1, 1 : -buffer_size] + \
                                                            roll_down[1 : -1, 1 : -buffer_size] + \
                                                            roll_left[1 : -1, 1 : -buffer_size] + \
                                                            roll_right[1 : -1, 1 : -buffer_size] - \
                                                            4 * read_layer[1 : -1, 1 : -buffer_size]

def laplacian_tl(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)

    # left and top
    # left top           
    write_layer[0, 0] = roll_up[0, 0 ] + roll_left[0, 0] - 2 * read_layer[0, 0 ]
    # left center
    write_layer[1 : -buffer_size, 0] = roll_down[1 : -buffer_size, 0] + roll_up[1 : -buffer_size, 0 ] + roll_left[1 : -buffer_size, 0] - 3 * read_layer[1: -buffer_size, 0]
    # center top
    write_layer[0, 1 : -buffer_size] = roll_up[0, 1 : -buffer_size] +  roll_left[0, 1 : -buffer_size] + roll_right[0, 1 : -buffer_size] - 3 * read_layer[0, 1 : -buffer_size]
    write_layer[1 : -buffer_size, 1 : -buffer_size] = roll_up[1 : -buffer_size, 1 : -buffer_size] + \
                                                            roll_down[1 : -buffer_size, 1 : -buffer_size] + \
                                                            roll_left[1 : -buffer_size, 1 : -buffer_size] + \
                                                            roll_right[1 : -buffer_size, 1 : -buffer_size] - \
                                                            4 * read_layer[1 : -buffer_size, 1 : -buffer_size]

def laplacian_trb(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)
    # right top
    write_layer[0, -1] = roll_up[0, -1 ] + roll_right[0, -1] - 2 * read_layer[0, -1]
    # right center
    write_layer[1 : -1, -1] = roll_up[1:-1, -1] + roll_down[1:-1, -1] + roll_right[1:-1, -1] - 3 * read_layer[1:-1, -1]
    # right bottom
    write_layer[-1, -1] = roll_down[-1, -1 ] + roll_right[-1, -1] - 2 * read_layer[-1, -1]

    # center top
    write_layer[0, buffer_size : -1] = roll_up[0, buffer_size : -1] +  roll_left[0, buffer_size : -1] + roll_right[0, buffer_size : -1] - 3 * read_layer[0, buffer_size : -1]
    # center bottom
    write_layer[-1, buffer_size : -1] = roll_down[-1, buffer_size : -1] +  roll_left[-1, buffer_size : -1] + roll_right[-1, buffer_size : -1] -3 * read_layer[-1, buffer_size : -1]
    write_layer[1 : -1, buffer_size : -1] = roll_up[1 : -1, buffer_size : -1] + \
                                                            roll_down[1 : -1, buffer_size : -1] + \
                                                            roll_left[1 : -1, buffer_size : -1] + \
                                                            roll_right[1 : -1, buffer_size : -1] - \
                                                            4 * read_layer[1 : -1, buffer_size : -1]

def laplacian_tr(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)

    # right top
    write_layer[0, -1] = roll_up[0, -1 ] + roll_right[0, -1] - 2 * read_layer[0, -1]
    # right center
    write_layer[1 : -buffer_size, -1] = roll_up[1:-buffer_size, -1] + roll_down[1:-buffer_size, -1] + roll_right[1:-buffer_size, -1] - 3 *  read_layer[1:-buffer_size, -1]
    # center top
    write_layer[0, buffer_size : -1] = roll_up[0, buffer_size : -1] +  roll_left[0, buffer_size : -1] + roll_right[0, buffer_size : -1] - 3 * read_layer[0, buffer_size : -1]

    write_layer[1 : -buffer_size, buffer_size : -1] = roll_up[1 : -buffer_size, buffer_size : -1] + \
                                                            roll_down[1 : -buffer_size, buffer_size : -1] + \
                                                            roll_left[1 : -buffer_size, buffer_size : -1] + \
                                                            roll_right[1 : -buffer_size, buffer_size : -1] - \
                                                            4 * read_layer[1 : -buffer_size, buffer_size : -1]

def laplacian_tb(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)
    # bottom and top unbuffered
    # center top
    write_layer[0, buffer_size : -buffer_size] = roll_up[0, buffer_size : -buffer_size] +  roll_left[0, buffer_size : -buffer_size] + roll_right[0, buffer_size : -buffer_size] - 3 * read_layer[0, buffer_size : -buffer_size]
    # center bottom
    write_layer[-1, buffer_size : -buffer_size] = roll_down[-1, buffer_size : -buffer_size] +  roll_left[-1, buffer_size : -buffer_size] + roll_right[-1, buffer_size : -buffer_size] - 3 * read_layer[-1, buffer_size : -buffer_size]
    write_layer[1 : -1, buffer_size : -buffer_size] = roll_up[1 : -1, buffer_size : -buffer_size] + \
                                                            roll_down[1 : -1, buffer_size : -buffer_size] + \
                                                            roll_left[1 : -1, buffer_size : -buffer_size] + \
                                                            roll_right[1 : -1, buffer_size : -buffer_size] - \
                                                            4 * read_layer[1 : -1, buffer_size : -buffer_size]

def laplacian_t(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)

    write_layer[0, buffer_size : -buffer_size] = roll_up[0, buffer_size : -buffer_size] +  roll_left[0, buffer_size : -buffer_size] + roll_right[0, buffer_size : -buffer_size] - 3 * read_layer[0, buffer_size : -buffer_size]
    write_layer[1 : -buffer_size, buffer_size : -buffer_size] = roll_up[1 : -buffer_size, buffer_size : -buffer_size] + \
                                                            roll_down[1 : -buffer_size, buffer_size : -buffer_size] + \
                                                            roll_left[1 : -buffer_size, buffer_size : -buffer_size] + \
                                                            roll_right[1 : -buffer_size, buffer_size : -buffer_size] - \
                                                            4 * read_layer[1 : -buffer_size, buffer_size : -buffer_size]

def laplacian_lrb(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)

    # left, right bottom unbuffered
    # left center
    write_layer[buffer_size : - 1, 0] = roll_down[buffer_size : -1, 0] + roll_up[buffer_size : -1, 0 ] + roll_left[buffer_size : -1, 0] - 3 * read_layer[buffer_size:-1, 0 ]
    # left bottom
    write_layer[-1, 0] = roll_down[-1, 0] + roll_left[-1, 0] - 2 * read_layer[-1, 0 ]
    # right center
    write_layer[buffer_size : -1, -1] = roll_up[buffer_size:-1, -1] + roll_down[buffer_size:-1, -1] + roll_right[buffer_size:-1, -1] - 3 * read_layer[buffer_size:-1, -1]
    # right bottom
    write_layer[-1, -1] = roll_down[-1, -1 ] + roll_right[-1, -1] - 2 * read_layer[-1, -1]

    # center bottom
    write_layer[-1, 1 : -1] = roll_down[-1, 1 : -1] +  roll_left[-1, 1 : -1] + roll_right[-1, 1 : -1] - 3 * read_layer[-1, 1 : -1]
    write_layer[buffer_size : -1, 1 : -1] = roll_up[buffer_size : -1, 1 : -1] + \
                                                            roll_down[buffer_size : -1, 1 : -1] + \
                                                            roll_left[buffer_size : -1, 1 : -1] + \
                                                            roll_right[buffer_size : -1, 1 : -1] - \
                                                            4 * read_layer[buffer_size : -1, 1 : -1]

def laplacian_lr(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)

     # left, right unbuffered
    # left center
    write_layer[buffer_size : -buffer_size, 0] = roll_down[buffer_size : -buffer_size, 0] + roll_up[buffer_size : -buffer_size, 0 ] + roll_left[buffer_size : -buffer_size, 0] - 3 * read_layer[buffer_size:-buffer_size, 0 ]
    # right center
    write_layer[buffer_size : -buffer_size, -1] = roll_up[buffer_size:-buffer_size, -1] + roll_down[buffer_size:-buffer_size, -1] + roll_right[buffer_size:-buffer_size, -1] - 3 * read_layer[buffer_size:-buffer_size, -1]
    write_layer[buffer_size : -buffer_size, 1 : -1] = roll_up[buffer_size : -buffer_size, 1 : -1] + \
                                                            roll_down[buffer_size : -buffer_size, 1 : -1] + \
                                                            roll_left[buffer_size : -buffer_size, 1 : -1] + \
                                                            roll_right[buffer_size : -buffer_size, 1 : -1] - \
                                                            4 * read_layer[buffer_size : -buffer_size, 1 : -1]

def laplacian_lb(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)

    # left bottom unbuffered
    # left center
    write_layer[buffer_size : - 1, 0] = roll_down[buffer_size : -1, 0] + roll_up[buffer_size: -1, 0 ] + roll_left[buffer_size : -1, 0] - 3 * read_layer[buffer_size:-1, 0 ]
    # left bottom
    write_layer[-1, 0] = roll_down[-1, 0] + roll_left[-1, 0] - 2 * read_layer[-1, 0 ]
    # center bottom
    write_layer[-1, 1 : -buffer_size] = roll_down[-1, 1 : -buffer_size] +  roll_left[-1, 1 : -buffer_size] + roll_right[-1, 1 : -buffer_size] - 3 * read_layer[-1, 1 : -buffer_size]
    write_layer[buffer_size : -1, 1 : -buffer_size] = roll_up[buffer_size : -1, 1 : -buffer_size] + \
                                                            roll_down[buffer_size : -1, 1 : -buffer_size] + \
                                                            roll_left[buffer_size : -1, 1 : -buffer_size] + \
                                                            roll_right[buffer_size : -1, 1 : -buffer_size] - \
                                                            4 * read_layer[buffer_size : -1, 1 : -buffer_size]

def laplacian_l(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)
    # left unbuffered
    # left center
    write_layer[buffer_size : -buffer_size, 0] = roll_down[buffer_size : -buffer_size, 0] + roll_up[buffer_size : -buffer_size, 0 ] + roll_left[buffer_size : -buffer_size, 0] - 3 * read_layer[buffer_size:-buffer_size, 0 ]
    write_layer[buffer_size : -buffer_size, 1 : -buffer_size] = roll_up[buffer_size : -buffer_size, 1 : -buffer_size] + \
                                                            roll_down[buffer_size : -buffer_size, 1 : -buffer_size] + \
                                                            roll_left[buffer_size : -buffer_size, 1 : -buffer_size] + \
                                                            roll_right[buffer_size : -buffer_size, 1 : -buffer_size] - \
                                                            4 * read_layer[buffer_size : -buffer_size, 1 : -buffer_size]

def laplacian_rb(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)

    # right, bottom unbuffered
    # right center
    write_layer[buffer_size : -1, -1] = roll_up[buffer_size:-1, -1] + roll_down[buffer_size:-1, -1] + roll_right[buffer_size:-1, -1] - 3 * read_layer[buffer_size:-1, -1]
    # right bottom
    write_layer[-1, -1] = roll_down[-1, -1 ] + roll_right[-1, -1] - 2 * read_layer[-1, -1]
    # center bottom
    write_layer[-1, buffer_size : -1] = roll_down[-1, buffer_size : -1] +  roll_left[-1, buffer_size : -1] + roll_right[-1, buffer_size : -1] - 3 * read_layer[-1, buffer_size : -1]
    write_layer[buffer_size : -1, buffer_size : -1] = roll_up[buffer_size : -1, buffer_size : -1] + \
                                                            roll_down[buffer_size : -1, buffer_size : -1] + \
                                                            roll_left[buffer_size : -1, buffer_size : -1] + \
                                                            roll_right[buffer_size : -1, buffer_size : -1] - \
                                                            4 * read_layer[buffer_size : -1, buffer_size : -1]

def laplacian_r(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)
    # right
    # right center
    write_layer[buffer_size : -buffer_size, -1] = roll_up[buffer_size:-buffer_size, -1] + roll_down[buffer_size:-buffer_size, -1] + roll_right[buffer_size:-buffer_size, -1] - 3 * read_layer[buffer_size:-buffer_size, -1]
    write_layer[buffer_size : -buffer_size, buffer_size : -1] = roll_up[buffer_size : -buffer_size, buffer_size : -1] + \
                                                            roll_down[buffer_size : -buffer_size, buffer_size : -1] + \
                                                            roll_left[buffer_size : -buffer_size, buffer_size : -1] + \
                                                            roll_right[buffer_size : -buffer_size, buffer_size : -1] - \
                                                            4 * read_layer[buffer_size : -buffer_size, buffer_size : -1]
    

def laplacian_b(read_layer, write_layer, buffer_size):
    roll_up = torch.roll(read_layer, -1, 0)
    # roll rows down
    roll_down = torch.roll(read_layer, 1, 0)
    #print(b)
    # roll cols left
    roll_left = torch.roll(read_layer, -1, 1)
    # roll cols right
    roll_right = torch.roll(read_layer, 1, 1)
    # bottom unbuffered
    # center bottom
    write_layer[-1, buffer_size : -buffer_size] = roll_down[-1, buffer_size : -buffer_size] + roll_left[-1, buffer_size : -buffer_size] + roll_right[-1, buffer_size : -buffer_size] - 3 * read_layer[-1, buffer_size : -buffer_size]
    write_layer[buffer_size : -1, buffer_size : -buffer_size] = roll_up[buffer_size : -1, buffer_size : -buffer_size] + \
                                                            roll_down[buffer_size : -1, buffer_size : -buffer_size] + \
                                                            roll_left[buffer_size : -1, buffer_size : -buffer_size] + \
                                                            roll_right[buffer_size : -1, buffer_size : -buffer_size] - \
                                                            4 * read_layer[buffer_size : -1, buffer_size : -buffer_size]

def get_laplacian_func(offsets):
    left_unbuffered = offsets[0] == 0
    right_unbuffered = offsets[1] == 0
    top_unbuffered = offsets[2] == 0
    bottom_unbuffered = offsets[3] == 0

    # 2D
    if np.all(offsets[:-2]):
        return laplacian

    elif top_unbuffered:
        # top is unbuffered
        if left_unbuffered:
            if right_unbuffered:
                if bottom_unbuffered:
                    return laplacian_tlrb
                else:
                    return laplacian_tlr
                    
            else:
                if bottom_unbuffered:
                    return laplacian_tlb
                    
                else:
                    return laplacian_tl
        else:
            # left buffered
            if right_unbuffered:
                if bottom_unbuffered:
                    return laplacian_trb
                    
                else:
                    return laplacian_tr
                    # right, top unbuffered
                    
            else:
                if bottom_unbuffered:
                    return laplacian_tb
                    
                else:
                    return laplacian_t
                   
    else:
        # top is buffered
        if left_unbuffered:
            if right_unbuffered:
                if bottom_unbuffered:
                    return laplacian_lrb

                else:
                    return laplacian_lr
                   
            else:
                if bottom_unbuffered:
                    return laplacian_lb
                   
                else:
                    return laplacian_l
        else:

            # left buffered
            if right_unbuffered:
                if bottom_unbuffered:
                    return laplacian_rb
                    
                else:
                    return laplacian_r
            else:
                if bottom_unbuffered:
                    return laplacian_b

    
class Diffuser:

    def __init__(self, rw_value_layer):
        self.laplacian = get_laplacian_func(rw_value_layer.read_layer.non_buff_grid_offsets)
    
    def __call__(self, rw_value_layer):
        self.laplacian(rw_value_layer.read_layer.impl.grid, 
                            rw_value_layer.write_layer.impl.grid,
                            rw_value_layer.read_layer.buffer_size)
        #print(rw_value_layer.write_layer.impl.grid)
        rw_value_layer.write_layer.impl.grid = rw_value_layer.read_layer.impl.grid + 0.1 * rw_value_layer.write_layer.impl.grid
            
