# %%
import numpy as np
np.set_printoptions(threshold=3000) #print out matching matrix in full
single_resource = ["ZXIXZI","IZXZII","IIZXZI","IIIZXZ","ZIIIZX","XZIIIZ"]
single_zeroes = np.zeros((6,6))

def main(dims):
    N = dims[0]*dims[1]*dims[2]
    M = construct_checkmatrix(N).astype(int)
    # print(M)
    # np.savetxt('dim333b.csv',M,delimiter=',')
    fuse_checkmatrix(dims,M)
    # np.savetxt('dim333a.csv',M,delimiter=',')
    stabilizers = np.array(deconstruct_checkmatrix(M)).astype(str)
    # np.savetxt('dim333_stabilizer.csv',stabilizers,delimiter=',',fmt='%s')

def construct_checkmatrix(dim):
    X,Z = create_checkXnZ(single_resource)
    X = np.array(X)
    Z = np.array(Z)
    zeroes = single_zeroes
    twodim = 2*dim
    for i in range(dim):
        for j in range(twodim):
            if j == 0 and i == 0:
                A = X
            elif j == 0 and i != 0:
                A = zeroes
            elif j == i:
                A = np.concatenate([A, X], axis=1)
            elif j == i+dim:
                A = np.concatenate([A, Z], axis=1)
            else:
                A = np.concatenate([A, zeroes], axis=1)
        if i == 0:
            B = A
        else:
            B = np.concatenate([B,A],axis=0)
    return B

def create_checkXnZ(sr):
    X = []
    Z = []
    for st in sr:
        x = []
        z = []
        if not st.isalpha():
            print("Please try to fix the stabilizer set.")
        else:
            num = len(st)
            for i in range(num):
                if st[i] == 'X':
                    x.append(1)
                    z.append(0)
                elif st[i] == 'Z':
                    x.append(0)
                    z.append(1)
                elif st[i] == 'Y':
                    x.append(1)
                    z.append(1)
                elif st[i] == 'I':
                    x.append(0)
                    z.append(0)
                else:
                    print("please fix stabilizer set")
        X.append(x)
        Z.append(z)
    return X,Z

def deconstruct_checkmatrix(Matrix):
    liststabilizer = []
    N,twoN = Matrix.shape
    for i in range(N):
        stabilizer = []
        for j in range(N):
            if Matrix[i][j] == 1 and Matrix[i][N+j] == 0:
                stabilizer.append('X')
            elif Matrix[i][j] == 0 and Matrix[i][N+j] == 1:
                stabilizer.append('Z')
            elif Matrix[i][j] == 1 and Matrix[i][N+j] == 1:
                stabilizer.append('Y')
            elif Matrix[i][j] == 0 and Matrix[i][N+j] == 0:
                stabilizer.append('I')
            else:
                print("something's wrong with the check matrix")
        liststabilizer.append(stabilizer)
    return liststabilizer

def cell_xytcoords_to_ix(dims, cell_xytcoords):
    return cell_xytcoords[0] + cell_xytcoords[1] * dims[0] \
            + cell_xytcoords[2] * dims[0] * dims[1]

def cell_ix_to_xytcoords(dims, cell_ix):
    z_coord = int(cell_ix / (dims[0] * dims[1]))
    y_coord = int(cell_ix / dims[0]) % dims[1]
    x_coord = cell_ix % dims[0]
    return np.array((x_coord, y_coord, z_coord))

def shifted_cell_ix(dims, cell_ix, shift, shift_axis):
    #     Function to get the index of a cell obtained shifting an initial cell with 'cell_ix'
    #     by a integer (positive or negative) step 'shift' along an axis 'shift_axis' in x, y, or t.
    if not isinstance(shift, int):
        raise ValueError('The parameter shift can only be an integer (positive or negative)')

    if shift_axis == 'x':
        axis_label = 0
        size_lim = dims[0] - 1
    elif shift_axis == 'y':
        axis_label = 1
        size_lim = dims[1] - 1
    elif shift_axis == 't':
        axis_label = 2
        size_lim = dims[2] - 1
    else:
        raise ValueError('Shift axis can only be one of (x, y, or t)')
    temp_coords = cell_ix_to_xytcoords(cell_ix)

    temp = (temp_coords[axis_label] + shift)
    # if temp <
    return cell_xytcoords_to_ix(temp_coords)

def fuse_1edge(matrix,rg):
    acommuting = matrix@(rg)%2
    rglen = len(rg)
    halflen = int(rglen/2)
    replace_arr = np.concatenate([rg[halflen:rglen],rg[0:halflen]])
    # print(acommuting)
    lgt = len(acommuting)
    flag = 0
    for i in range(lgt):
        if flag == 0 and acommuting[i] == 1:
            flag = 1
            temp = matrix[i].copy()
            matrix[i] = replace_arr
            # print(temp)
        elif flag == 1 and acommuting[i] == 1:
            matrix[i] = matrix[i] ^ temp
            # print(matrix[i])
    if not flag:
        print(rg)
        raise ValueError("Check matrix all commuting")

def fuse_checkmatrix(dims,M):
    N = dims[0]*dims[1]*dims[2]
    zeroX = np.zeros(N*6)
    for i in range(N):
        coord = cell_ix_to_xytcoords(dims,i)
        x_coord_next = coord[0] + 1
        y_coord_next = coord[1] + 1
        z_coord_next = coord[2] + 1
        if z_coord_next < dims[2]:
            temp_coord = coord.copy()
            temp_coord[2] = z_coord_next
            next_z = cell_xytcoords_to_ix(dims,temp_coord)
            temp_z = zeroX.copy()
            temp_z[i*6] = 1
            temp_z[next_z*6+3] = 1
            temp = np.concatenate([zeroX,temp_z])
            fuse_1edge(M,temp)
        if y_coord_next < dims[0]:
            temp_coord = coord.copy()
            temp_coord[1] = y_coord_next
            next_y = cell_xytcoords_to_ix(dims,temp_coord)
            temp_y = zeroX.copy()
            temp_y[i*6+1] = 1
            temp_y[next_y*6+4] = 1
            temp = np.concatenate([zeroX,temp_y])
            fuse_1edge(M,temp)
        if x_coord_next < dims[0]:
            temp_coord = coord.copy()
            temp_coord[0] = x_coord_next
            next_x = cell_xytcoords_to_ix(dims,temp_coord)
            temp_x = zeroX.copy()
            temp_x[i*6+2] = 1
            temp_x[next_x*6+5] = 1
            temp = np.concatenate([zeroX,temp_x])
            fuse_1edge(M,temp)

# %%
if __name__ == '__main__':
    dim = [10,10,10]
    # dim = [2,2,2]
    main(dim)
# %%
