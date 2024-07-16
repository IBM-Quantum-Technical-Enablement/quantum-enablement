#--------------------------------------------------
# Lattice module -- AFI branch
# Created: 05/23/2024
#---------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from numpy import logical_and as AND
from numpy import logical_or as OR
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

# Number of qubits within the unit cell
unit_cell_size  = {}
unit_cell_size['heavy_hex'] = 12
unit_cell_size['heavy_hex_flat'] = 12

# Graph degree of the lattice
graph_degree  = {}
graph_degree['heavy_hex'] = 3
graph_degree['heavy_hex_flat'] = 3

unit_cell_descr = {}
unit_cell_descr['heavy_hex'] = {}
unit_cell_descr['heavy_hex']['top'] = [2]
unit_cell_descr['heavy_hex']['right'] = [5]
unit_cell_descr['heavy_hex']['bottom'] = [8]
unit_cell_descr['heavy_hex']['left'] = [11]

num_layers = {}
num_layers['heavy_hex'] = 3

def unit_cell_colorings(tp,conf):

    if tp in ['heavy_hex','heavy_hex_flat'] and conf == 'regular':

        val = np.zeros([3,2,12],int)
        val[0,0] = [1,2,3,1,3,1,2,1,3,2,3,2]
        val[1,0] = [2,3,1,2,1,2,3,2,1,3,1,3]
        val[2,0] = [3,1,2,3,2,3,1,3,2,1,2,1]
        val[0,1] = [1,2,3,1,3,2,3,2,1,3,1,2]
        val[1,1] = [2,3,1,2,1,3,1,3,2,1,2,1]
        val[2,1] = [3,1,2,3,2,1,2,1,3,2,3,1]

    if tp in ['heavy_hex','heavy_hex_flat'] and conf == 'AFI':

        val = np.zeros([3,2,12],int)
        val[0,0] = [3,2,3,1,2,1,2,3,1,3,1,2]
        val[1,0] = [3,2,3,1,2,1,2,3,1,3,1,2]
        val[2,0] = [3,2,3,1,2,1,2,3,1,3,1,2]
        val[0,1] = [3,2,3,1,2,1,2,3,1,3,1,2]
        val[1,1] = [3,2,3,1,2,1,2,3,1,3,1,2]
        val[2,1] = [3,2,3,1,2,1,2,3,1,3,1,2]

    return val

# Positions of individual qubits within a unit cell
def unit_cell_vertices(tp):

    vec = {}

    if tp == 'heavy_hex':
        vec[0]  = [-2, 2]
        vec[1]  = [-1, 3]
        vec[2]  = [ 0, 4]
        vec[3]  = [ 1, 3]
        vec[4]  = [ 2, 2]
        vec[5]  = [ 2, 0]
        vec[6]  = [ 2,-2]
        vec[7]  = [ 1,-3]
        vec[8]  = [ 0,-4]
        vec[9]  = [-1,-3]
        vec[10] = [-2,-2]
        vec[11] = [-2, 0]

    return vec

# Positions of the center of the unit cell
def unit_cell_positions(tp,indx1,indx2):

    if tp == 'heavy_hex':
        return 4*indx1 + 2*(indx2%2),-6*indx2
    
    if tp == 'heavy_hex_flat':
        return 4*indx1 + 2*(indx2%2),-2*indx2
    
def custom_lattice_cut(size,qubits):

    lattice = lattice_2d(tp = 'heavy_hex', nx = size[0], ny = size[1], conf = 'regular')
    qlist = [q for q in np.arange(lattice.n_qubits) if q not in qubits]

    # remove qubits
    lattice.remove_qubits(qlist)

    return lattice

class lattice_2d:

    def __init__(self, tp = 'heavy_hex', nx = 0, ny = 0, device_name = '', conf = 'AFI', mps_order=None):

        self.layers = num_layers[tp]
        self.tp = tp

        if tp in ['heavy_hex','heavy_hex_flat']:

            # lattice size parameters
            self.nx = nx
            self.ny = ny

            #-----------------------------------------
            # Derive qubit positions
            #-----------------------------------------

            # TARGET: position of individual qubits
            self.positions = {}
            # dynamical qubit index
            q_index = 0

            # dictionary that identify qubits by their positions
            find_qubit = {}

            # collection of unit cells
            self.lattice = {}

            # positions of individual qubits within the unit cell
            ucv = unit_cell_vertices(tp)
            colorings = unit_cell_colorings(tp,conf)

            # in a loop, run over all unit cells
            for j in range(ny):
                for i in range(nx):
                
                    # unit cell defined by qubit indices
                    unit_cell = {}
                    
                    # positions of the center of the unit_cell
                    x0,y0 = unit_cell_positions(tp, i, j)
                
                    for k in range(unit_cell_size[tp]):

                        # position of individual qubits of the unit cell
                        x = x0 + ucv[k][0]
                        y = y0 + ucv[k][1]

                        # if the qubits already belongs to some other unit cell,
                        # just put its index
                        if (x,y) in find_qubit:
                            unit_cell[k] = find_qubit[(x,y)]

                        # if the qubits does not belong to existing cell
                        # add new index and add its coordinates to the list
                        # of positions
                        else:
                            unit_cell[k] = q_index
                            find_qubit[(x,y)] = q_index
                            self.positions[q_index] = [x,y]
                            q_index += 1
                    
                    # add the cell to the lattice
                    self.lattice[(i,j)] = unit_cell.copy()

            self.n_qubits = q_index
            #-----------------------------------------
            # Find the lattice connectivity
            #-----------------------------------------

            # TARGET: list of all connectivities with their order in
            # Floquet sequence
            self.couplings = {}

            # in the loop, assign the connectivities within each unit cell
            for i in range(nx):
                for j in range(ny):

                    # define the unit cell
                    unit_cell = self.lattice[(i,j)]

                    for k in range(unit_cell_size[tp]):

                        # get the indices of the neighboring qubits
                        indx1,indx2 = unit_cell[k],unit_cell[(k+1)%unit_cell_size[tp]]

                        # if connectivity does not recorded, add it
                        if not ((indx1,indx2) in self.couplings or (indx2,indx1) in self.couplings):
                            self.couplings[(indx1,indx2)] = colorings[i%3,j%2][k]
                            self.couplings[(indx2,indx1)] = colorings[i%3,j%2][k]
            
            if mps_order is None:
                mps_position_sort = sorted(self.positions.values(), key=lambda x: (reversor(x[1]), x[0]))
                self.mps_to_qubit = {i:find_qubit[tuple(mps_position_sort[i])] for i in range(len(mps_position_sort))}
                self.qubit_to_mps = {v:k for (k,v) in self.mps_to_qubit.items()}
            else:
                self.mps_to_qubit = {i:mps_order[i] for i in range(len(mps_order))}
                self.qubit_to_mps = {v:k for (k,v) in self.mps_to_qubit.items()}

        
        self.original_numbering = {}
        self.original_to_current = {}
        for q in range(self.n_qubits):
           self.original_numbering[q] = q
           self.original_to_current[q] = q


        self.original_numbering = {}
        self.original_to_current = {}
        for q in range(self.n_qubits):
           self.original_numbering[q] = q
           self.original_to_current[q] = q

    def mps_to_lat_idx(self, x):
        return self.mps_to_qubit[x]
    
    def lat_to_mps_idx(self, x):
        return self.qubit_to_mps[x]
    
    def draw(self,show_gates = False, show_mps_order = False, qubit_assignment = [],cmap = 'Greys',
             enumerate_qubits=False, original_numbering = False, show = True, return_canva = False, 
             save=False, path=None, range_var = None,draw_current = False,current = [],
             margin = (0.1,0.1),assign_links = {}, ):
        
        q,s = 0.5*np.sqrt(3), 0.5
            
        # if no values assigned to qubits, do not coloring
        if len(qubit_assignment) == 0:
            qubit_assignment = [0]*self.n_qubits

        # set the lattice dependent plot rescaling
        #fig,ax = plt.subplots()
            
        xvals = [self.positions[i][0] for i in self.positions]
        yvals = [self.positions[i][1] for i in self.positions]

        dx = np.max(xvals)-np.min(xvals)
        dy = np.max(yvals)-np.min(yvals)

        if self.tp in ['heavy_hex']:
            fig,ax = plt.subplots(figsize = (0.75*dx,0.8/2*dy))

        if self.tp in ['heavy_hex_flat','device']:
            fig,ax = plt.subplots(figsize = (0.75*dx,1.6/2*dy))

        # remove axes
        ax.set_axis_off()

        # if gate coloring requested, assign colors to edges
        if show_gates:
            col = ['k','b','g','r']

        # else, make them all the same
        else:
            col = ['k','k','k','k']

        # add vertices
        x_vals = np.zeros(len(self.positions))
        y_vals = np.zeros(len(self.positions))
        for indx in self.positions:
            x,y = self.positions[indx]
            x_vals[indx] = q*x
            y_vals[indx] = s*y

        if range_var==None:
            vmax,vmin = np.max(qubit_assignment),np.min(qubit_assignment)
        if isinstance(range_var,list):
            vmin,vmax = range_var

        ax.scatter(x_vals,y_vals,s=500,zorder=1,c=qubit_assignment,ec='k',cmap=cmap,vmax = vmax, vmin=vmin)
        

            # add numbers to vertices
        if enumerate_qubits:
            for indx in self.positions:
                x,y = self.positions[indx]
            
                if indx!=0:
                    if self.tp in ['heavy_hex']:
                        if original_numbering:
                            indx = self.original_numbering[indx]
                        ax.text(q*x-0.08-0.05*np.round(np.log(indx)/np.log(10),0),s*y-0.05,str(indx),color='k')
                    if self.tp in ['heavy_hex_flat','device']:
                        if original_numbering:
                            indx = self.original_numbering[indx]
                        ax.text(q*x-0.08-0.05*np.round(np.log(indx)/np.log(10),0),s*y-0.05,str(indx),color='k')
                else:
                    if self.tp in ['heavy_hex']:
                        if original_numbering:
                            indx = self.original_numbering[indx]
                        ax.text(q*x-0.08,s*y-0.1,str(indx),color='k')
                    if self.tp in ['heavy_hex_flat','device']:
                        if original_numbering:
                            indx = self.original_numbering[indx]
                        ax.text(q*x-0.08,s*y-0.05,str(indx),color='k')

        # add edges
        if len(assign_links)==0:
            for link in self.couplings:
                indx1,indx2 = link
                x1,y1 = self.positions[indx1]
                x2,y2 = self.positions[indx2]

                ax.plot([q*x1,q*x2],[s*y1,s*y2],zorder=0,lw=5,c = col[self.couplings[link]])

                if draw_current:

                    m = abs(current[indx1,indx2])
                    if current[indx1,indx2]>0:
                        ax.arrow(q*x1,s*y1, 0.5*q*(x2-x1), 0.5*s*(y2-y1), lw = m,head_width=0.3*m, head_length=0.2*m, fc='k', ec='k')
                    if current[indx1,indx2]<0:
                        ax.arrow(q*x2,s*y2, -0.5*q*(x2-x1), -0.5*s*(y2-y1), lw = m,head_width=0.3*m, head_length=0.2*m, fc='k', ec='k')
            
        else:

            # Define the RGB values for green and red
            green = np.array([0, 1, 0])
            red = np.array([1, 0, 0])

            # Create a list of colors, interpolating between green and red
            colors = [green * (1 - i) + red * i for i in np.linspace(0, 1, 256)]

            # Create a LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("green_to_red", colors)

            vals = [assign_links[key] for key in assign_links]
            norm = LogNorm(vmin=np.min(vals), vmax=np.max(vals))
            #mcolors.Normalize(vmin=np.min(vals), vmax=np.max(vals))

            for link in self.couplings:
                indx1,indx2 = link
                x1,y1 = self.positions[indx1]
                x2,y2 = self.positions[indx2]

                ax.plot([q*x1,q*x2],[s*y1,s*y2],zorder=0,lw=5,c = cmap(norm(assign_links[link])))

            # Add a colorbar to the right of the plot
            mappable = ScalarMappable(norm=norm, cmap=cmap)

            # Add the colorbar using the ScalarMappable
            fig.colorbar(mappable, ax=ax)

        ax.set_xmargin(margin[0])
        ax.set_ymargin(margin[1])
            
        #ax.set_xlim(np.min(xvals)-1,np.max(xvals)+1)
        #ax.set_ylim(np.min(yvals)-0.5,np.max(yvals)+0.5)
        
        if show_mps_order:
            pos = np.array([self.positions[self.mps_to_qubit[i]] for i in range(self.n_qubits)])
            ax.plot(q*pos[:, 0], s*pos[:, 1], linestyle=':', linewidth=2.)
            for i, p in enumerate(pos):
                ax.text(q*p[0]+0.2, s*p[1]+0.2, str(i))

        if save:
            fig.savefig(path + ".svg")
        if show:
            plt.show()
        if return_canva:
            return ax 
        
    def modify_layout(self, todo, qlist):

        # remove duplicates
        qlist = np.unique(qlist).tolist()

        # determine surviving qubits
        if todo == 'remove':
            surviors = [i for i in range(self.n_qubits) if i not in qlist]
        if todo == 'keep':
            surviors = qlist

        original_numbering_new = {}

        # set the positions of surviving qubits
        positions_new = {}
        for q in self.positions:
            if q in surviors:
                positions_new[surviors.index(q)] = self.positions[q]
                original_numbering_new[surviors.index(q)] = self.original_numbering[q]

        # set the couplings of surviving qubits
        couplings_new = {}
        for pair in self.couplings:
            q0,q1 = pair
            if q0 in surviors and q1 in surviors:
                couplings_new[(surviors.index(q0),surviors.index(q1))] = self.couplings[pair]

        # replace `original` positions with one for surviving qubits
        self.positions = positions_new.copy()
        self.couplings = couplings_new.copy()
        self.original_numbering = original_numbering_new.copy()

        self.original_to_current = {}
        for q in self.original_numbering:
            self.original_to_current[self.original_numbering[q]] = q
            
        # reduce the number of qubits recorded
        if todo == 'remove':
            self.n_qubits += -len(qlist)
        if todo == 'keep':
            self.n_qubits = len(qlist)


class reversor:
    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        return other.obj == self.obj

    def __lt__(self, other):
        return other.obj < self.obj