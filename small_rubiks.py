import itertools
import numpy as np
from task import task, setup

def turn_cube(old, axis, layer):
    
    #face order: left, right, bottom, top, front, back

    if axis == 0:
        A = (1,0,0,0,0,-1,0,1,0)
        rot = (0,1,4,5,3,2)
        #top->back->bottom->front->top, right and left stay in place
    elif axis == 1:
        A = (0,0,-1, 0,1,0, 1,0,0)
        rot = (4,5,2,3,1,0)
        #right->back->left->front->right, top and bottom stay in place
    elif axis == 2:
        A = (0,-1,0, 1,0,0, 0,0,1)
        rot = (2,3,1,0,4,5)
        #right->top->left->bottom->right, front and back stay in place
    else:
        raise Exception()

    a_11,a_12,a_13,a_21,a_22,a_23,a_31,a_32,a_33 = A

    new = np.zeros((3,3,3,6),dtype=int)

    for x,y,z,f in itertools.product([-1,0,1],[-1,0,1],[-1,0,1],range(6)):

        if (x,y,z)[axis] == layer:
            x_=a_11*x+a_12*y+a_13*z
            y_=a_21*x+a_22*y+a_23*z
            z_=a_31*x+a_32*y+a_33*z
            f_=rot[f]
        else:
            x_,y_,z_,f_=x,y,z,f
        
        new[x_,y_,z_,f_]=old[x,y,z,f]

    return new

i=0
visilist = []
for x,y,z in itertools.product([-1,1],[-1,1],[-1,1]):
    if x==-1:
        visilist.append((x,y,z,0,i))
        i+=1
    if x==1:
        visilist.append((x,y,z,1,i))
        i+=1
    if y==-1:
        visilist.append((x,y,z,2,i))
        i+=1
    if y==1:
        visilist.append((x,y,z,3,i))
        i+=1
    if z==-1:
        visilist.append((x,y,z,4,i))
        i+=1
    if z==1:
        visilist.append((x,y,z,5,i))
        i+=1

identity = -np.ones((3,3,3,6),dtype=int)
for (x,y,z,f,i) in visilist:
    identity[x,y,z,f] = i

turns = []

for a,l in itertools.product([0,1,2],[-1,1]):
    turns.append(turn_cube(identity,a,l))

np_turns = []
for t in turns:
    #Add reverse turns?
    nt = np.zeros(len(visilist),dtype=int)
    for (x,y,z,f,i) in visilist:
        if t[x,y,z,f] == -1:
            raise Exception()
        nt[i] = t[x,y,z,f]

    np_turns.append(nt)
    np_turns.append(nt[nt[nt]])

np_turns.append(np.arange(24))
np_turns = np.array(np_turns)

#Perform a manual rotation of a specified layer around a specified axis 
def manual_turn(cube,axis,layer,reverse=False):
    if reverse:
        return cube[np_turns[4*axis+(layer+1)+1]]
    else:
        return cube[np_turns[4*axis+(layer+1)]]

start_coloring = -np.ones(len(visilist),dtype=int)
for (_,__,___,f,i) in visilist:
    start_coloring[i] = f

if np.max(start_coloring == -1) > 0:
    raise Exception()

faces = -np.ones((6,2,2),dtype=int)

for i,j in itertools.product([-1,1],[-1,1]):

    #We think about looking at the cube from the front first, rotating around the y-axis to get to right, left and back faces,
    #and rotating around the x-axis to get to the top and bottom faces.
    k = (i+1)//2
    l = (j+1)//2
    faces[4,k,l] = identity[i,j,-1,4]
    faces[0,k,l] = identity[-1,j,-i,0]
    faces[5,k,l] = identity[-i,j,1,5]
    faces[1,k,l] = identity[1,j,i,1]
    faces[2,k,l] = identity[i,-1,-j,2]
    faces[3,k,l] = identity[i,1,j,3]

if np.max(faces == -1) > 0:
    raise Exception()

def get_colors(flat_colors):

    colors = -np.ones((6,2,2),dtype=int)
    for f,i,j in itertools.product(range(6),range(2),range(2)):
        colors[f,i,j] = flat_colors[faces[f,i,j]]

    if np.max(colors == -1) > 0:
        raise Exception()

    return colors

mesh_positions = []
for i,j in itertools.product(range(2),range(2)):
    #each tuple has the form: (x,y, f,i,j) where x and y are the coordinates where the color should be drawn,
    #and f,i,j say from which small face the color should be taken
    mesh_positions.append((i,2+j, 0,i,j))
    mesh_positions.append((4+i,2+j, 1,i,j))
    mesh_positions.append((2+i,j, 2,i,j))
    mesh_positions.append((2+i,4+j, 3,i,j))
    mesh_positions.append((2+i,2+j, 4,i,j))
    mesh_positions.append((6+i,2+j, 5,i,j))

#Outputs the mesh of a cube on the console
def print_coloring(cube_state, color_array=False):

    if color_array:
        colors = get_colors(cube_state)
    else:
        colors = get_colors(start_coloring[cube_state])

    grid = np.zeros((8,6),dtype=int)
    for x,y,f,i,j in mesh_positions:
        grid[x,y] = colors[f,i,j]+1

    letters = [" ","R","O","G","B","W","Y"]
    text=""
    for y in range(6):
        for x in range(8):
            text += letters[grid[x,5-y]]
        text += "\n"

    print(text)

winning_states = []

s1 = np.array(range(24),dtype=int)

def turn_all_layers(state, axis, reverse = False):

    for i in [-1,1]:
        state = manual_turn(state,axis=axis, layer=i, reverse=reverse)

    return state

for _ in range(4):
    s1 = turn_all_layers(s1,axis=1)
    s2=s1
    
    for __ in range(4):
        s2 = turn_all_layers(s2,axis=0)
        winning_states.append(start_coloring[s2])

    s2=turn_all_layers(s1,axis=2)
    winning_states.append(start_coloring[s2])
    s2=turn_all_layers(s1,axis=2,reverse=True)
    winning_states.append(start_coloring[s2])

compare = np.zeros((24,24),dtype=bool)
for i,j in itertools.product(range(24),range(24)):
    if i!= j:
        compare[i,j] = (winning_states[i] == winning_states[j]).all()

#Call this function on a set of cubes to see which of them are finished
def check_win(state):

    colors = start_coloring[state]

    for w in winning_states:
        if (colors==w).all():
            return True

    return False

#Call this function to turn the cube. Code 6a+2l turns layer l of axis a 
#in mathematically positive direction when looking on the fingertip in a left-handed coordinate system.
#Code 6a+2l+1 turns in reverse direction, and code -1 leaves the cube as before. 

def task_action(state, action_code):
    
    return True, state[np_turns[action_code]]

# def task_action(states, action_codes):

#     n,columns = states.shape
#     if columns != 54:
#         raise Exception("Each cube must have length 54")
    
#     turns = np_turns[action_codes]
#     return np.take_along_axis(states, turns, axis = 1)

def make_neural_input(state):

    colors = start_coloring[state]
    input = np.zeros((24,6),dtype=float)
    input[np.arange(24,dtype=int),colors] = 1.0
    return input.flatten()

gamma = .5

def reward_function(state, steps):

    if check_win(state):
        return 1 - gamma+ gamma/steps
    else:
        return 0

rubiks_task = task(12,task_action,check_win, reward_function)
rubiks_setup = setup(np.arange(24),12,task_action)